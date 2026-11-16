# 项目规则（Project Rules）

## 密钥与敏感信息安全规范

1. **严禁提交任何密钥信息**：不得将任何 API Key、Token、密码、私钥（如 `sk-...`、`.env`、`.pem`、`.key` 等）写入代码、配置文件、README、示例模板、注释、日志或提交记录中。
2. **仅使用本地 `.env` 存放凭据**：所有真实密钥必须放在项目根目录的 `.env` 文件中（该文件已被 `.gitignore` 忽略），仅允许将不含真实密钥的 `.env.example` 模板纳入版本控制。
3. **提交前必须检查**：每次执行 `git add` / `git commit` / `git push` 前，必须核查 `git status` 与 `git diff --cached`，确认不包含 `.env` 或任何疑似密钥字符串（如 `sk-`、`Bearer `、`API_KEY=` 后的真实值）；一旦发现，立即中止提交并清除敏感内容。
