import pymysql, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control')
df = pd.read_sql("SELECT model_version, true_label, pred_label FROM model_predictions", conn)
conn.close()

bert_df = df[df['model_version'] == 'BERT']
textcnn_df = df[df['model_version'] == 'TextCNN']

def evaluate(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['安全', '有害'], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(report).transpose(), cm

report_t, cm_t = evaluate(textcnn_df['true_label'], textcnn_df['pred_label'])
report_b, cm_b = evaluate(bert_df['true_label'], bert_df['pred_label'])

fig = plt.figure(figsize=(16, 12))
fig.suptitle('有害内容检测双模型对比评估报告', fontsize=20, fontweight='bold')

# 混淆矩阵
ax1 = fig.add_subplot(2,3,1)
sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['安全','有害'], yticklabels=['安全','有害'])
ax1.set_title('TextCNN 混淆矩阵'); ax1.set_xlabel('预测'); ax1.set_ylabel('真实')

ax2 = fig.add_subplot(2,3,2)
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['安全','有害'], yticklabels=['安全','有害'])
ax2.set_title('BERT 混淆矩阵'); ax2.set_xlabel('预测'); ax2.set_ylabel('真实')

# 指标表
ax3 = fig.add_subplot(2,3,3); ax3.axis('off')
table_data = report_t[['precision','recall','f1-score']].iloc[:2].round(4).values
ax3.table(cellText=table_data, colLabels=['Precision','Recall','F1'], rowLabels=['安全','有害'],
          cellLoc='center', loc='center')
ax3.set_title('TextCNN 分类指标')

ax4 = fig.add_subplot(2,3,4); ax4.axis('off')
table_data_b = report_b[['precision','recall','f1-score']].iloc[:2].round(4).values
ax4.table(cellText=table_data_b, colLabels=['Precision','Recall','F1'], rowLabels=['安全','有害'],
          cellLoc='center', loc='center')
ax4.set_title('BERT 分类指标')

# 准确率柱状图
ax5 = fig.add_subplot(2,3,5)
acc_t = (textcnn_df['true_label'] == textcnn_df['pred_label']).mean()
acc_b = (bert_df['true_label'] == bert_df['pred_label']).mean()
bars = ax5.bar(['TextCNN','BERT'], [acc_t, acc_b], color=['#2E86AB','#A23B72'])
ax5.set_ylim(0.9, 1.0); ax5.set_title('双模型测试准确率对比')
for bar, acc in zip(bars, [acc_t, acc_b]):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f'{acc:.4f}', ha='center')

# 训练曲线（真实数据替换）
ax6 = fig.add_subplot(2,3,6)
tcnn_epochs = [1, 2, 3]
tcnn_train = [0.8750, 0.9100, 0.9320]   # 替换为实际值
tcnn_test  = [0.8321, 0.8750, 0.8900]
bert_epochs = [1, 2]
bert_train = [0.9963, 0.9978]   # 替换为实际值
bert_test  = [0.9942, 0.9932]
ax6.plot(tcnn_epochs, tcnn_train, 'b-o', label='TextCNN Train')
ax6.plot(tcnn_epochs, tcnn_test, 'b--o', label='TextCNN Test')
ax6.plot(bert_epochs, bert_train, 'r-o', label='BERT Train')
ax6.plot(bert_epochs, bert_test, 'r--o', label='BERT Test')
ax6.set_xlabel('Epoch'); ax6.set_ylabel('Accuracy')
ax6.legend(); ax6.set_title('训练准确率变化曲线')

plt.tight_layout()
plt.savefig('/home/wcy/nlp-risk-control/images/comprehensive_report.png', dpi=150, bbox_inches='tight')
print("综合评估报告已保存至 images/comprehensive_report.png")
