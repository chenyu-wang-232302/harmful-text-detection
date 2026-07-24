# 敏感垃圾评论识别器 | 有害文本检测
## 📌 项目简介
实现中文有害文本二分类任务，识别广告、色情、谩骂、博彩、欺诈类垃圾敏感文本。
基于 ChineseHarm-Bench 公开数据集，对比传统机器学习与深度学习方案：
1. TF-IDF + SVM（传统机器学习）
2. TextCNN 文本卷积神经网络
3. BERT-base-chinese 微调

## 🛠️ 技术栈
Python、PyTorch、Transformers、Scikit-learn、Streamlit、Pandas

## 📊 数据集
ChineseHarm-Bench 中文有害文本数据集
数据集地址：https://github.com/zjunlp/ChineseHarm-bench
原始6分类任务，转换为二分类：
- label=1：有害文本（色情、辱骂、博彩、广告、诈骗）
- label=0：正常合规文本

## 🚀 快速运行
1. 安装依赖
```bash
pip install -r requirements.txt
