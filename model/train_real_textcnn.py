import pymysql, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba, random, numpy as np, json

# ===== 1. 从 MySQL 读取数据 =====
conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control')
cursor = conn.cursor()
cursor.execute("SELECT original_text, true_label FROM model_predictions")
data = cursor.fetchall()
conn.close()

texts = [row[0] for row in data]
labels = [row[1] for row in data]

print(f"总样本数: {len(texts)}")
print(f"有害样本: {sum(labels)}, 安全样本: {len(labels)-sum(labels)}")

# ===== 2. 构建词汇表 =====
word_count = {}
for text in texts:
    for word in jieba.lcut(text):
        word_count[word] = word_count.get(word, 0) + 1

vocab = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:2000]
word_to_idx = {word: idx+1 for idx, (word, _) in enumerate(vocab)}
vocab_size = len(word_to_idx) + 1

def text_to_sequence(text, max_len=32):
    seq = [word_to_idx.get(word, 0) for word in jieba.lcut(text)]
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [0] * (max_len - len(seq))

# ===== 3. 数据集类 =====
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        seq = text_to_sequence(self.texts[idx])
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

random.seed(42)
indices = list(range(len(texts)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]

train_dataset = TextDataset([texts[i] for i in train_idx], [labels[i] for i in train_idx])
test_dataset = TextDataset([texts[i] for i in test_idx], [labels[i] for i in test_idx])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===== 4. TextCNN 模型 =====
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_classes=2):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (2, embed_dim)),
            nn.Conv2d(1, 100, (3, embed_dim)),
            nn.Conv2d(1, 100, (4, embed_dim))
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = TextCNN(vocab_size, num_classes=2)

# ===== 5. 类别权重 =====
pos_count = sum(labels)
neg_count = len(labels) - pos_count
weight_for_0 = len(labels) / (2.0 * neg_count)
weight_for_1 = len(labels) / (2.0 * pos_count)
class_weights = torch.tensor([weight_for_0, weight_for_1])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# ===== 6. 训练与验证 =====
def train_epoch():
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = (all_preds == all_targets).mean()
    miss = ((all_targets == 1) & (all_preds == 0)).sum() / max((all_targets == 1).sum(), 1)
    fp = ((all_targets == 0) & (all_preds == 1)).sum() / max((all_targets == 0).sum(), 1)
    return acc, miss, fp

print("开始训练 TextCNN...")
for epoch in range(3):
    loss = train_epoch()
    train_acc, _, _ = evaluate(train_loader)
    test_acc, miss, fp = evaluate(test_loader)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Miss={miss:.4f}, FP={fp:.4f}")

# ===== 7. 保存模型和词汇表 =====
torch.save(model.state_dict(), '/home/wcy/nlp-risk-control/model/textcnn_real.pth')
with open('/home/wcy/nlp-risk-control/model/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(word_to_idx, f, ensure_ascii=False)
print("模型已保存")

# ===== 8. 对全量数据预测并更新 MySQL（model_version='TextCNN'） =====
model.eval()
conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control')
cursor = conn.cursor()
cursor.execute("SELECT post_id, original_text FROM model_predictions WHERE model_version='TextCNN'")
rows = cursor.fetchall()

for post_id, text in rows:
    seq = torch.tensor([text_to_sequence(text)], dtype=torch.long)
    with torch.no_grad():
        outputs = model(seq)
        prob = torch.softmax(outputs, dim=1)[0, 1].item()
    pred_label = 1 if prob > 0.5 else 0
    cursor.execute("UPDATE model_predictions SET pred_label=%s, pred_prob=%s WHERE post_id=%s",
                   (pred_label, prob, post_id))

conn.commit()
conn.close()
print("TextCNN 预测结果已更新 MySQL")
