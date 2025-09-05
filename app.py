
import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

def has_key():
    v = os.getenv("OPENAI_API_KEY") or ""
    return bool(v) and v.strip().startswith("sk-")

def get_client():
    if not has_key():
        raise RuntimeError("OPENAI_API_KEY is missing or invalid. Set it in Railway → Variables.")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def health():
    return jsonify({"ok": True, "service": "artemo-openai-backend"})

@app.get("/diag")
def diag():
    return jsonify({"has_openai_key": has_key()})

@app.get("/envkeys")
def envkeys():
    keys = []
    for k, v in os.environ.items():
        if k.startswith("OPENAI") or k.startswith("RAILWAY") or k in ("PORT","PYTHON_VERSION"):
            if k == "OPENAI_API_KEY" and v:
                masked = f"{v[:3]}***len={len(v)}"
            else:
                masked = str(v)
            keys.append({"key": k, "value_preview": masked})
    keys.sort(key=lambda x: x["key"])
    return jsonify({"env": keys})

@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image")
    mime = data.get("mime") or "image/jpeg"
    if not image_b64:
        return jsonify({"error": "image required"}), 400

    prompt = "你是一名艺术治疗从业者（非医疗诊断）。请根据用户上传的画作，输出严格 JSON。"

    try:
        client = get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是严谨的 JSON 生成器。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
                    ]
                }
            ],
            temperature=0.4,
        )
        content = resp.choices[0].message.content.strip()
        try:
            return jsonify(json.loads(content))
        except Exception:
            return jsonify({"analysis": content})
    except Exception as e:
        return jsonify({"error": "AI request failed", "detail": str(e)}), 502

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.logger.info("Startup diag: has_key=%s, OPENAI_API_KEY len=%s",
                    has_key(), len(os.getenv("OPENAI_API_KEY") or ""))
    app.run(host="0.0.0.0", port=port, debug=True)
