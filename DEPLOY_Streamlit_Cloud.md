# 使用 Streamlit Community Cloud 部署 ML Textbook Chatbot

本应用已具备部署所需文件，按以下步骤即可在 Streamlit Cloud 上运行。

---

## 一、前置条件

1. **GitHub 账号**  
   - 若没有：到 [github.com](https://github.com) 注册。

2. **代码在 GitHub 仓库中**  
   - Streamlit Cloud 只支持从 **GitHub 仓库** 部署，不支持本地或其它 Git 托管。

3. **API 密钥（应用内输入即可）**  
   - 本应用在侧边栏要求用户输入 **OpenAI API Key** 和 **Pinecone API Key**，部署后用户自行填写即可，**无需** 在 Cloud 里配置 Secrets（除非你想预填默认值）。

---

## 二、把项目推送到 GitHub

### 1. 在项目目录初始化 Git（若尚未初始化）

```bash
cd /Users/wangbiliu/Documents/26winter-EEP596A/miniproject_2
git init
```

### 2. 添加并提交文件

```bash
git add app.py agents.py requirements.txt .streamlit/ .gitignore
git commit -m "Add Streamlit Cloud deployment support"
```

**注意：** 不要提交 `.streamlit/secrets.toml`、`.env` 或任何包含真实 API Key 的文件（已写在 `.gitignore` 中）。

### 3. 在 GitHub 上新建仓库并推送

- 打开 [github.com/new](https://github.com/new)。
- 填写仓库名（例如 `miniproject_2` 或 `ml-textbook-chatbot`），选择 **Public**。
- **不要** 勾选 “Add a README”等（本地已有代码）。
- 创建后，按页面提示执行（替换成你的用户名和仓库名）：

```bash
git remote add origin https://github.com/你的用户名/你的仓库名.git
git branch -M main
git push -u origin main
```

若已有 `remote`，只需：

```bash
git push -u origin main
```

---

## 三、在 Streamlit Community Cloud 上部署

### 1. 打开 Streamlit Cloud

- 浏览器访问：**https://share.streamlit.io**  
- 点击 **“Sign up”** 或 **“Log in”**，选择 **“Sign in with GitHub”**，按提示授权。

### 2. 创建新应用

- 登录后进入你的 **Workspace**。
- 点击右上角 **“New app”**（或 “Create app”）。

### 3. 填写应用配置

在 “Deploy an app” 页面：

| 选项 | 填写说明 |
|------|----------|
| **Repository** | 选择你刚推送的仓库，例如 `你的用户名/miniproject_2` |
| **Branch** | 一般为 `main`（或你推送的分支名） |
| **Main file path** | 填写 `app.py`（入口文件） |

- 若提示 “Do you have an app?” 选 **“Yes, I have an app”**，然后按上表填写。

### 4. 高级设置（可选）

点击 **“Advanced settings”** 可设置：

- **Python version**：建议 3.11 或 3.12（默认即可）。
- **Secrets**：本应用在侧边栏输入 API Key，通常**不需要**在这里填。若你希望预填默认 Key（不推荐把真实 Key 写在仓库里），可在此粘贴 TOML，例如：

```toml
# 仅作示例，不建议把真实 Key 写在 Cloud Secrets 里
OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "..."
```

保存后不填也可以，用户在使用时在网页侧边栏输入即可。

### 5. 部署

- 点击 **“Deploy!”**。
- 等待 2–5 分钟，Cloud 会拉取代码、安装 `requirements.txt` 并启动 `streamlit run app.py`。
- 完成后会显示应用链接，形如：  
  `https://你的应用名.streamlit.app`

---

## 四、部署后使用方式

1. 打开 Cloud 给你的链接。
2. 在左侧 **Configuration** 中填写：
   - **OpenAI API Key**
   - **Pinecone API Key**
   - **Pinecone Index Name**（默认 `machine-learning-textbook`）
   - **Namespace**（默认 `ns2500`）
3. 填写完成后即可在输入框提问，使用 ML 课本 RAG 对话。

---

## 五、项目里已为部署准备的文件

| 文件 | 作用 |
|------|------|
| `requirements.txt` | Cloud 据此安装 Python 依赖（streamlit、openai、pinecone、langchain 等） |
| `.streamlit/config.toml` | Streamlit 基础配置（CORS、统计等） |
| `.gitignore` | 避免把 `secrets.toml`、`.env`、虚拟环境等提交到 GitHub |

---

## 六、常见问题

**Q: 部署一直转圈 / 报错？**  
- 查看 Cloud 页面上的 **Logs**，常见原因：  
  - `requirements.txt` 里缺包或版本不兼容 → 在本地用 `pip install -r requirements.txt` 试跑一次。  
  - 仓库里没有 `app.py` 或路径填错 → 确认 Main file path 为 `app.py`。

**Q: 需要固定 Python 版本？**  
- 在 **Advanced settings** 里选择 Python 3.11 或 3.12。

**Q: 更新代码后如何生效？**  
- 把改动 `git push` 到同一分支，Streamlit Cloud 会自动重新部署（有时需等 1–2 分钟）。

**Q: 能否用私有仓库？**  
- Streamlit Community Cloud 支持私有仓库，但可能需在授权时允许其访问该私有仓库。

按上述步骤即可完成从本机到 Streamlit Cloud 的部署；应用本身不依赖本地路径，只要 GitHub 上有 `app.py`、`agents.py` 和 `requirements.txt` 即可运行。
