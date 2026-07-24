import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
BERT_NAME = "bert-base-chinese"

# ====================== 1. 数据集处理模块 ======================
def load_data():
    """
    使用ChineseHarm-Bench数据集，数据集原始6分类，转为二分类：
    有害类别(Pornography/Abuse/Gambling/Ads/Fraud)=1，正常Non-Violation=0
    文件格式：csv 两列 text, category
    """
    df = pd.read_csv("ChineseHarm.csv")
    harm_categories = ["Pornography", "Abuse", "Gambling", "Illicit Ads", "Fraud"]
    df["label"] = df["category"].apply(lambda x: 1 if x in harm_categories else 0)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    return train_test_split(texts, labels, test_size=0.2, random_state=42)

# ====================== 2. 传统机器学习 TF-IDF + SVM ======================
def run_tfidf_svm(X_train, X_test, y_train, y_test):
    print("===== 开始训练 TF-IDF + SVM 模型 =====")
    tfidf = TfidfVectorizer()
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)

    svm = SVC()
    svm.fit(X_train_tf, y_train)
    y_pred = svm.predict(X_test_tf)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"SVM 准确率：{acc:.4f}, F1分数：{f1:.4f}")
    return {"acc": acc, "f1": f1, "model": svm, "tfidf": tfidf}

# ====================== 3. TextCNN 模型定义与训练 ======================
class SimpleTokenizer:
    @staticmethod
    def encode(s, truncation, max_length, padding):
        vec = [vocab.get(c, 1) for c in list(str(s))][:max_length]
        if len(vec) < max_length:
            vec += [0] * (max_length - len(vec))
        return vec

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        seq = self.tokenizer.encode(self.texts[idx], truncation=True, max_length=MAX_LEN, padding="max_length")
        return torch.tensor(seq), torch.tensor(self.labels[idx])

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, 128, k) for k in [2,3,4]])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(3*128, num_classes)
    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        out = [torch.max_pool1d(torch.relu(c), c.size(-1)).squeeze(-1) for c in self.convs]
        out = torch.cat(out, dim=1)
        return self.fc(self.dropout(out))

def run_textcnn(X_train, X_test, y_train, y_test):
    print("===== 开始训练 TextCNN 模型 =====")
    # 构建字符词表
    from collections import Counter
    all_text = X_train + X_test
    all_chars = []
    for s in all_text:
        all_chars.extend(list(str(s)))
    counter = Counter(all_chars)
    global vocab
    vocab = {"<PAD>":0, "<UNK>":1}
    for char, _ in counter.most_common():
        vocab[char] = len(vocab)

    train_set = TextDataset(X_train, y_train, SimpleTokenizer)
    test_set = TextDataset(X_test, y_test, SimpleTokenizer)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = TextCNN(vocab_size=len(vocab)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    epoch = 2
    history_acc = []

    for e in range(epoch):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = loss_fn(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # 测试
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                logits = model(bx)
                pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred_idx)
                y_true.extend(by.numpy())
        acc = accuracy_score(y_true, y_pred)
        history_acc.append(acc)
        print(f"TextCNN Epoch{e+1} Acc:{acc:.4f}")
    f1 = f1_score(y_true, y_pred)
    return {"acc": history_acc[-1], "f1": f1, "history": history_acc, "model": model}

# ====================== 4. BERT模型定义与训练 ======================
class BertDataSet(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        res = self.tok(text, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": res["input_ids"][0],
            "attention_mask": res["attention_mask"][0],
            "label": torch.tensor(label)
        }

class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:,0,:]
        return self.fc(self.drop(cls_vec))

def run_bert(X_train, X_test, y_train, y_test):
    print("===== 开始微调 BERT 模型 =====")
    tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    train_set = BertDataSet(X_train, y_train, tokenizer)
    test_set = BertDataSet(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)

    model = BertClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=2e-5)
    epoch = 2
    history_acc = []

    for e in range(epoch):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab = batch["label"].to(device)
            pred = model(ids, mask)
            loss = loss_fn(pred, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # evaluate
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(ids, mask)
                pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred_idx)
                y_true.extend(batch["label"].numpy())
        acc = accuracy_score(y_true, y_pred)
        history_acc.append(acc)
        print(f"BERT Epoch{e+1} Acc:{acc:.4f}")
    f1 = f1_score(y_true, y_pred)
    return {"acc": history_acc[-1], "f1": f1, "history": history_acc, "model": model, "tokenizer": tokenizer}

# ====================== 5. Streamlit可视化界面 ======================
def run_streamlit(results_cnn, results_bert, results_svm):
    import streamlit as st
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    st.set_page_config(page_title="敏感/垃圾评论识别器", layout="wide")
    st.title("敏感词/垃圾评论识别器｜ChineseHarm-Bench")

    st.subheader("模型指标汇总")
    table = pd.DataFrame({
        "模型":["TF-IDF+SVM","TextCNN","BERT"],
        "准确率":[results_svm["acc"], results_cnn["acc"], results_bert["acc"]],
        "F1分数":[results_svm["f1"], results_cnn["f1"], results_bert["f1"]]
    })
    st.dataframe(table)

    st.subheader("训练收敛曲线")
    fig, ax = plt.subplots()
    ax.plot([1,2], results_cnn["history"], marker="o", label="TextCNN")
    ax.plot([1,2], results_bert["history"], marker="o", label="BERT")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("测试集准确率")
    ax.legend()
    st.pyplot(fig)

    st.subheader("在线预测测试")
    input_text = st.text_input("输入评论文本：", value="带你稳赚30万，博彩内部渠道")
    if st.button("检测"):
        tok = results_bert["tokenizer"]
        model = results_bert["model"]
        model.eval()
        encode = tok(input_text, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        ids = encode["input_ids"].to(device)
        mask = encode["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(ids, mask)
            pred = torch.argmax(logits, dim=1).item()
        if pred == 1:
            st.error("检测结果：有害/垃圾敏感文本")
        else:
            st.success("检测结果：正常文本")

# ====================== 程序入口 ======================
if __name__ == "__main__":
    print("加载数据集...")
    X_train, X_test, y_train, y_test = load_data()

    # 依次运行三类模型
    result_svm = run_tfidf_svm(X_train, X_test, y_train, y_test)
    result_cnn = run_textcnn(X_train, X_test, y_train, y_test)
    result_bert = run_bert(X_train, X_test, y_train, y_test)

    print("\n===== 全部模型训练完成 =====")
    print(f"SVM        Acc:{result_svm['acc']:.4f} F1:{result_svm['f1']:.4f}")
    print(f"TextCNN    Acc:{result_cnn['acc']:.4f} F1:{result_cnn['f1']:.4f}")
    print(f"BERT       Acc:{result_bert['acc']:.4f} F1:{result_bert['f1']:.4f}")

    # 启动可视化界面
    run_streamlit(result_cnn, result_bert, result_svm)
