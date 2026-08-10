import pymysql, torch, torch.nn as nn, jieba, json, numpy as np

with open('/home/wcy/nlp-risk-control/model/vocab.json', 'r') as f:
    word_to_idx = json.load(f)
vocab_size = len(word_to_idx) + 1

def text_to_sequence(text, max_len=32):
    seq = [word_to_idx.get(word, 0) for word in jieba.lcut(text)]
    if len(seq) > max_len: return seq[:max_len]
    return seq + [0]*(max_len-len(seq))

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1,100,(2,embed_dim)), nn.Conv2d(1,100,(3,embed_dim)), nn.Conv2d(1,100,(4,embed_dim))
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = self.dropout(torch.cat(x, dim=1))
        return self.fc(x)

model = TextCNN(vocab_size, num_classes=2)
model.load_state_dict(torch.load('/home/wcy/nlp-risk-control/model/textcnn_real.pth', map_location='cpu'))
model.eval()

conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control')
cursor = conn.cursor()
cursor.execute("SELECT post_id, original_text FROM model_predictions WHERE model_version='TextCNN'")
rows = cursor.fetchall()

for post_id, text in rows:
    seq = torch.tensor([text_to_sequence(text)], dtype=torch.long)
    with torch.no_grad():
        outputs = model(seq)
        prob = torch.softmax(outputs, dim=1)[0,1].item()
    pred_label = 1 if prob > 0.5 else 0
    cursor.execute("UPDATE model_predictions SET pred_label=%s, pred_prob=%s WHERE post_id=%s", (pred_label, prob, post_id))

conn.commit()
conn.close()
print("TextCNN 预测完成")
