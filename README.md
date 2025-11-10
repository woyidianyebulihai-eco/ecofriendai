# EcoHome Advisor — 入门版（零基础也能部署）

## 本地运行（Windows/Mac）
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 在 Streamlit Cloud 部署
1. 把 `app.py` 和 `requirements.txt` 上传到你的 GitHub 仓库根目录（新建仓库也可以）。
2. 打开 https://share.streamlit.io/ ，点击 **New app**。
3. 选你的仓库、分支（main），**Main file path** 填 `app.py`，点击 **Deploy**。
4. 等待构建完成，页面就上线了（会给你一个公开链接）。
