import json, os, re, pymysql

conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control', charset='utf8mb4')
cursor = conn.cursor()

label_map = {"不违规":0,"低俗色情":1,"博彩":1,"欺诈":1,"谩骂引战":1,"黑产广告":1}

def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.!?;:()（）]', '', text)
    text = ' '.join(text.split())
    return text

data_dir = '/home/wcy/nlp-risk-control/data/raw'
for filename in os.listdir(data_dir):
    if not filename.endswith('.json'): continue
    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
        records = json.load(f)
        for idx, item in enumerate(records):
            text = item['文本']
            label = label_map.get(item['标签'], -1)
            cleaned = clean_text(text)
            post_id = f"{filename}_{idx}"
            cursor.execute("INSERT INTO model_predictions (post_id, original_text, true_label, pred_label, pred_prob, model_version) VALUES (%s,%s,%s,-1,0.0,'pending')", (post_id, cleaned, label))

conn.commit()
conn.close()
print("清洗入库完成")
