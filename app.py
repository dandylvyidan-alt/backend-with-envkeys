
import os, json, time
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from openai import APIStatusError

app = Flask(__name__)
CORS(app)

def _has_key():
    v = (os.getenv("OPENAI_API_KEY") or "").strip()
    return bool(v) and v.startswith("sk-")

def get_client():
    if not _has_key():
        raise RuntimeError("OPENAI_API_KEY is missing or invalid. Set it in Railway → Variables.")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60)

@app.get("/")
def health():
    return jsonify({"ok": True, "service": "artemo-openai-backend-pro"})

@app.get("/diag")
def diag():
    return jsonify({"has_openai_key": _has_key()})

@app.get("/envkeys")
def envkeys():
    keys = []
    for k, v in os.environ.items():
        if k.startswith("OPENAI") or k.startswith("RAILWAY") or k in ("PORT","PYTHON_VERSION"):
            masked = f"{(v[:3]+'***') if v else ''}len={len(v) if v else 0}" if k=="OPENAI_API_KEY" else str(v)
            keys.append({"key": k, "value_preview": masked})
    keys.sort(key=lambda x: x["key"])
    return jsonify({"env": keys})

MAX_RETRIES = 3
def call_openai_with_retry(client, messages, model="gpt-4o-mini"):
    for i in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=900
            )
        except APIStatusError as e:
            if e.status_code in (429, 500, 502, 503, 504) and i < MAX_RETRIES - 1:
                time.sleep(1.2 * (i + 1))
                continue
            raise
        except Exception:
            raise

@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image")
    mime = (data.get("mime") or "image/jpeg").strip()
    model = (data.get("model") or "gpt-4o-mini").strip()
    if not image_b64:
        return jsonify({"error": "image required"}), 400
    if len(image_b64) > 1_800_000:
        return jsonify({"error": "image too large, please compress", "hint":"Compress to <700KB"}), 413

    system = (
        "You are an art therapy-informed assistant. You DO NOT provide diagnosis. "
        "You analyze drawings using formal elements (color, line, space/composition), symbols & narrative, "
        "process inferences, and structured projective drawing frameworks (HTP, DAP, KFD, mandala), "
        "then propose gentle, skills-based interventions from Rubin/Kramer/Wadeson. "
        "Respect cultural context; include uncertainty; keep language supportive and tentative; "
        "return STRICT JSON ONLY."
    )

    schema = {
      "summary":"string",
      "formal_analysis":{
        "color":{"observations":"string","affect_links":[{"pattern":"string","reference":["Luscher 1971","Wadeson 1980"]}]},
        "line":{"observations":"string","affect_links":[{"pattern":"string","inference":"string"}]},
        "space_composition":{"observations":"string","safety_control_inference":"string","reference":["Betensky 1995"]},
        "evidence":["string"]
      },
      "symbolic_analysis":{
        "motifs":[{"item":"string","possible_meanings":["string"],"confidence":0.0,"sources":["Jung 1964","Kellogg 1978","Hammer 1997"]}],
        "narrative":"string"
      },
      "process_inference":{"likely_medium":"string","gesture_marks":"string","effort_tolerance":"string","limits":"string"},
      "psychometrics":{
        "htp":{"house":"string","tree":"string","person":"string","caveat":"string"},
        "dap":{"body_parts":"string","posture":"string","caveat":"string"},
        "kfd":{"interactions":"string","boundaries":"string","caveat":"string"},
        "mandala_stage":{"stage_candidates":["string"],"why":"string","caveat":"string"}
      },
      "emotions":[{"label":"string","score":0,"evidence":"string"}],
      "treatment_plan":{
        "goals_short_term":["string"],
        "goals_long_term":["string"],
        "sessions":[{"title":"string","method":"string","materials":["string"],"steps":["string"],"duration_min":0,"when_to_use":"string"}],
        "home_practice":["string"]
      },
      "cultural_considerations":"string",
      "risk":{"flag":False,"signals":["string"],"action":"string"},
      "meta":{"uncertainty":0.0,"disclaimer":"string","citations":["Rubin 2010","Kramer 1971","Wadeson 1980","Lowenfeld & Brittain 1970","Betensky 1995","Buck 1948","Machover 1949","Burns & Kaufman 1970","Kellogg 1978","Jung 1964","Hammer 1997","Luscher 1971"]}
    }

    user_text = '''
严格遵循以下分析顺序并只输出 JSON：
1) 形式与构图（颜色/线条/空间-构图）：结合 Lüscher(1971)、Wadeson(1980)、Betensky(1995) 的思路，先观察，再给“可能的情绪联系”，避免绝对化。
2) 图像与象征：识别常见意象（树/房/人/门窗/太阳/月亮/路/动物等），参考 Jung(1964)、Kellogg(1978)、Hammer(1997) 给出“可能含义+置信度”与多义性。
3) 过程推断：根据笔触与媒材，谨慎推断控制/放松与耐受度的线索，说明局限。
4) 心理测量框架（非诊断、仅作结构化观察）：
   - HTP(Buck 1948)：house/tree/person 分项观察
   - DAP(Machover 1949)：人像的比例/姿态/部位
   - KFD(Burns & Kaufman 1970)：互动与边界
   - Mandala(Kellogg 1978)：原型阶段候选及理由
5) 情绪向量（0-100）与证据句。
6) 治疗计划：基于 Rubin/Kramer/Wadeson 的低风险、可复制干预（会话内 20-30 分钟），包含目标/步骤/材料/适用时机；附家庭练习。
7) 文化与伦理：Moon(2002) 风险避免与跨文化敏感性一句。
8) 风险提示：如存在自伤/伤人符号等，仅标记“需进一步评估”，给非紧急求助建议。
9) Meta：不确定性（0-1），声明“不是临床诊断”。

严格返回如下 JSON 模式（示例字段见 schema），不要加任何解释性文字。若证据不足，字段留空或以 caveat 说明。
'''

    try:
        client = get_client()
        messages = [
            {"role":"system", "content": system},
            {"role":"user", "content":[
                {"type":"text","text": json.dumps({"schema":schema}, ensure_ascii=False)},
                {"type":"text","text": user_text},
                {"type":"image_url","image_url":{"url": f"data:{mime};base64,{image_b64}"}}
            ]}
        ]
        resp = call_openai_with_retry(client, messages, model=model)
        content = (resp.choices[0].message.content or "").strip()
        try:
            return jsonify(json.loads(content))
        except Exception:
            return jsonify({"analysis_text": content, "meta":{"disclaimer":"模型返回了非JSON文本，已原样附上；此输出仅供一般性参考，不构成诊断。"}})
    except APIStatusError as e:
        status = getattr(e, "status_code", 502)
        detail = "OpenAI 429（额度或速率限制）" if status==429 else str(e)
        return jsonify({"error":"AI request failed", "detail": detail}), status
    except Exception as e:
        return jsonify({"error":"AI request failed", "detail": str(e)}), 502

if __name__ == "__main__":
    port = int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
