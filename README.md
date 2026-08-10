# 全链路智能内容风控分析平台| 有害文本检测
## 项目背景
某社交平台每天产生百万级 UGC 内容，原有关键词审核系统存在高误杀和高漏放问题，用户投诉量激增。本项目旨在引入 NLP 深度学习模型，对比 TextCNN 与 BERT 在有害内容检测上的性能差异，搭建从数据采集、实时流处理、模型训练评估到 A/B 实验分析的全链路数据平台，为业务决策提供数据支持。

### 中文有害文本数据集 
数据集地址：https://github.com/zjunlp/ChineseHarm-bench 原始6分类任务，转换为二分类：
label=1：有害文本（色情、辱骂、博彩、广告、诈骗）
label=0：正常合规文本

## 技术栈
- **数据采集**：Python 多源 JSON 文件解析，MySQL 存储
- **实时管道**：Kafka + PySpark Structured Streaming（备用批处理方案）
- **模型训练**：PyTorch 构建 TextCNN，Transformers 微调 BERT
- **存储与查询**：MySQL 业务库，Elasticsearch 指标存储（可选）
- **可视化**：Python Matplotlib 生成评估报告，Tableau 制作归因分析看板
- **实验管理**：MySQL A/B 实验分流表，SQL 指标计算

## 数据架构
[多源 JSON 日志] → [Python 清洗入库] → [MySQL ODS 层]
↓
[Kafka 模拟实时流] ← [Python 生产者] ← [MySQL 查询]
↓
[PySpark Streaming 消费] → [模型预测] → [MySQL 结果表]
↓（备用）
[Python 批处理] → [TextCNN/BERT 预测] → [MySQL]
↓
[SQL 评估指标] → [Tableau 报表] / [Matplotlib 综合报告]
## 指标体系
| 指标名称 | 计算逻辑 | 业务含义 |
|----------|----------|----------|
| 漏放率 (Miss Rate) | 有害样本被预测为安全的比例 | 社区安全风险 |
| 误杀率 (False Positive Rate) | 安全样本被预测为有害的比例 | 用户体验伤害 |
| 精确率 (Precision) | 预测为有害中真正有害的比例 | 审核准确度 |
| 召回率 (Recall) | 真实有害中被正确识别的比例 | 有害拦截能力 |
| F1-Score | 精确率与召回率的调和平均 | 综合性能 |
| 推理延迟 | 单条文本平均推理时间 | 线上成本 |

## 快速开始
### 1. 环境准备
- Ubuntu 20.04+，Python 3.10，Java 11/17
- 安装依赖：`pip install -r requirements.txt`
- 启动 MySQL：`sudo systemctl start mysql`
- （可选）启动 Kafka：`/opt/kafka/bin/kafka-server-start.sh -daemon /opt/kafka/config/server.properties`

### 2. 数据清洗入库
cd consumer
python3 clean_and_store.py

### 3. 模型训练与预测
TextCNN：
cd model
python3 train_real_textcnn.py
BERT 微调：
python3 finetune_bert_quick.py

### 4. 生成评估报告
python3 generate_evaluation_report.py

### 5. A/B 实验分流
USE risk_control;
INSERT INTO ab_experiment_log (post_id, model_group)
SELECT post_id, CASE WHEN RAND() < 0.3 THEN 'experiment' ELSE 'control' END
FROM model_predictions;

### 核心成果展示
双模型对比指标
模型	准确率	漏放率	误杀率	推理延迟
TextCNN	98.23%	0.48%	6.00%	~3ms
BERT	99.32%	0.70%	0.53%	~20ms
综合评估报告
https://images/comprehensive_report.png

### 实验结论与业务决策
结论：BERT 在语义理解上具有显著优势，几乎不漏放有害内容，且对安全内容的误伤极低；TextCNN 推理速度更快，适合高并发场景，但误杀率偏高。
决策：建议采用“分层审核”策略——高风险内容（新用户、短文本、敏感 IP）使用 BERT 进行精准审核，普通内容使用 TextCNN 快速过滤。通过 30% 灰度流量验证后，全量上线可使整体误杀率降低至 1% 以下，人工审核成本减少 40%。

### 后续优化方向
引入 GPU 进行全参数 BERT 微调，进一步提升精度
部署 TensorFlow Serving 或 TorchServe 实现线上推理
使用 Flink 替代 PySpark Streaming，提升实时性
引入模型可解释性工具（SHAP/LIME）分析误判案例















