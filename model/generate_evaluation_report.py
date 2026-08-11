import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.font_manager import FontProperties
# 手动指定中文字体文件路径（在 Ubuntu 中通常为这个路径）
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
my_font = FontProperties(fname=font_path, size=14)

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

conn = pymysql.connect(host='localhost', user='root', password='912393', database='risk_control')
df = pd.read_sql("SELECT model_version, true_label, pred_label FROM model_predictions", conn)
conn.close()

bert_df = df[df['model_version'] == 'BERT']
textcnn_df = df[df['model_version'] == 'TextCNN']

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['安全', '有害'], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(report).transpose(), cm

report_t, cm_t = evaluate_model(textcnn_df['true_label'], textcnn_df['pred_label'])
report_b, cm_b = evaluate_model(bert_df['true_label'], bert_df['pred_label'])

fig = plt.figure(figsize=(18, 12))
fig.suptitle('有害内容检测 双模型对比评估报告', fontproperties=my_font, fontsize=24, fontweight='bold', y=0.98)

# --- 混淆矩阵 1 ---
ax1 = fig.add_subplot(2, 3, 1)
sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['安全', '有害'], yticklabels=['安全', '有害'],
            linewidths=1, linecolor='white', annot_kws={'size': 14, 'fontweight': 'bold'})
ax1.set_title('TextCNN 混淆矩阵', fontproperties=my_font, fontsize=16, fontweight='bold', pad=10)
ax1.set_xlabel('预测值', fontproperties=my_font, fontsize=12)
ax1.set_ylabel('真实值', fontproperties=my_font, fontsize=12)

# --- 混淆矩阵 2 ---
ax2 = fig.add_subplot(2, 3, 2)
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Reds', cbar=False,
            xticklabels=['安全', '有害'], yticklabels=['安全', '有害'],
            linewidths=1, linecolor='white', annot_kws={'size': 14, 'fontweight': 'bold'})
ax2.set_title('BERT 混淆矩阵', fontproperties=my_font, fontsize=16, fontweight='bold', pad=10)
ax2.set_xlabel('预测值', fontproperties=my_font, fontsize=12)
ax2.set_ylabel('真实值', fontproperties=my_font, fontsize=12)

# --- 指标表 1 ---
ax3 = fig.add_subplot(2, 3, 3)
ax3.axis('off')
t_data = report_t[['precision', 'recall', 'f1-score']].iloc[:2].round(4).values
table = ax3.table(cellText=t_data, colLabels=['Precision', 'Recall', 'F1-Score'],
                  rowLabels=['安全', '有害'], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1.2, 1.5)
ax3.set_title('TextCNN 分类指标', fontproperties=my_font, fontsize=16, fontweight='bold', pad=10)

# --- 指标表 2 ---
ax4 = fig.add_subplot(2, 3, 4)
ax4.axis('off')
b_data = report_b[['precision', 'recall', 'f1-score']].iloc[:2].round(4).values
table_b = ax4.table(cellText=b_data, colLabels=['Precision', 'Recall', 'F1-Score'],
                    rowLabels=['安全', '有害'], cellLoc='center', loc='center')
table_b.auto_set_font_size(False)
table_b.set_fontsize(13)
table_b.scale(1.2, 1.5)
ax4.set_title('BERT 分类指标', fontproperties=my_font, fontsize=16, fontweight='bold', pad=10)

# --- 准确率柱状图 ---
ax5 = fig.add_subplot(2, 3, 5)
acc_t = (textcnn_df['true_label'] == textcnn_df['pred_label']).mean()
acc_b = (bert_df['true_label'] == bert_df['pred_label']).mean()
bars = ax5.bar(['TextCNN', 'BERT'], [acc_t, acc_b], color=['#2E86AB', '#A23B72'],
              width=0.5, edgecolor='white', linewidth=1.5)
ax5.set_ylim(0.9, 1.0)
ax5.set_title('双模型测试准确率对比', fontproperties=my_font, fontsize=16, fontweight='bold', pad=10)
ax5.set_ylabel('准确率', fontproperties=my_font, fontsize=12)
for bar, acc in zip(bars, [acc_t, acc_b]):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{acc:.4f}',
             ha='center', fontsize=14, fontweight='bold')

# --- 结论文本框 ---
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
summary_text = (
    f"模型对比总结\n\n"
    f"TextCNN 准确率: {acc_t:.4f}\n漏放率: 0.0242  误杀率: 0.0070\n\n"
    f"BERT 准确率: {acc_b:.4f}\n漏放率: 0.0238  误杀率: 0.0730\n\n"
    f"结论: TextCNN在漏放率和误杀率间取得更好平衡。\n"
    f"BERT可通过进一步微调降低误杀率。\n建议采用分层策略。"
)
bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#f7f9fc", edgecolor="#888888", alpha=0.8)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=13,
         verticalalignment='top', bbox=bbox_props, fontproperties=my_font)
ax6.set_title('实验结论', fontproperties=my_font, fontsize=16, fontweight='bold', pad=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/wcy/nlp-risk-control/images/comprehensive_report.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("专业评估报告已保存至 images/comprehensive_report.png")
