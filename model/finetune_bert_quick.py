import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pymysql, torch, torch.nn as nn, torch.optim as optim, numpy as np, random
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

# 加载分词器和BERT主体
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert = BertModel.from_pretrained('bert-base-chinese')

# 冻结所有BERT层，只训练后续的分类头（这是加速的关键）
for param in bert.parameters():
    param.requires_grad = False

class BertClassifierHead(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_emb))

model = BertClassifierHead(bert)

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# 从MySQL读取数据
conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control')
cursor = conn.cursor()
cursor.execute("SELECT original_text, true_label FROM model_predictions")
data = cursor.fetchall()
conn.close()

texts, labels = [r[0] for r in data], [r[1] for r in data]
random.seed(42)
indices = list(range(len(texts)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_dataset = BERTDataset([texts[i] for i in indices[:split]], [labels[i] for i in indices[:split]], tokenizer)
test_dataset  = BERTDataset([texts[i] for i in indices[split:]], [labels[i] for i in indices[split:]], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 类别权重
pos_count = sum(labels)
neg_count = len(labels) - pos_count
weight_for_0 = len(labels) / (2.0 * neg_count)
weight_for_1 = len(labels) / (2.0 * pos_count)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([weight_for_0, weight_for_1]))
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

def evaluate(loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            all_preds.extend(torch.argmax(outputs, dim=1).tolist())
            all_targets.extend(batch['label'].tolist())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = (all_preds == all_targets).mean()
    miss = ((all_targets == 1) & (all_preds == 0)).sum() / max((all_targets == 1).sum(), 1)
    fp   = ((all_targets == 0) & (all_preds == 1)).sum() / max((all_targets == 0).sum(), 1)
    return acc, miss, fp

# 训练2个epoch
for epoch in range(2):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
    train_acc, _, _ = evaluate(train_loader)
    test_acc, miss, fp = evaluate(test_loader)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Miss={miss:.4f}, FP={fp:.4f}")

# 保存模型
torch.save(model.state_dict(), '/home/wcy/nlp-risk-control/model/bert_quick_finetuned.pth')

# 全量预测并更新MySQL中model_version='BERT'的记录
model.eval()
conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control')
cursor = conn.cursor()
cursor.execute("SELECT post_id, original_text FROM model_predictions WHERE model_version='BERT'")
rows = cursor.fetchall()
for post_id, text in rows:
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
    with torch.no_grad():
        outputs = model(enc['input_ids'], enc['attention_mask'])
        prob = torch.softmax(outputs, dim=1)[0, 1].item()
    pred_label = 1 if prob > 0.5 else 0
    cursor.execute("UPDATE model_predictions SET pred_label=%s, pred_prob=%s WHERE post_id=%s", (pred_label, prob, post_id))
conn.commit()
conn.close()
print("BERT 快速微调完成，预测结果已更新到 MySQL")

