import pymysql
import json
import time
import random
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
)

conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control')
cursor = conn.cursor()
cursor.execute("SELECT post_id, original_text, true_label FROM model_predictions")
rows = cursor.fetchall()

for row in rows:
    msg = {'post_id': row[0], 'original_text': row[1], 'true_label': row[2]}
    producer.send('raw_content', value=msg)
    time.sleep(random.uniform(0.1, 0.5))

conn.close()
producer.flush()
print("所有数据已发送至 Kafka")
