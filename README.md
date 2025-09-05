
# Backend with /diag and /envkeys

Endpoints:
- `/` health
- `/diag` → {"has_openai_key": true/false}
- `/envkeys` → list OPENAI*/RAILWAY* env vars (masked)

Deploy:
1) Upload all files to GitHub repo root
2) Railway → New Project → Deploy from GitHub Repo (or Redeploy existing)
3) Service → Variables (Environment=Production) → add OPENAI_API_KEY=sk-...
4) Redeploy, then open https://<domain>/envkeys and /diag
