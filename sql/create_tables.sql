CREATE DATABASE IF NOT EXISTS risk_control CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE risk_control;

CREATE TABLE IF NOT EXISTS model_predictions (
    post_id VARCHAR(64) PRIMARY KEY,
    original_text TEXT NOT NULL,
    true_label INT NOT NULL COMMENT '0:安全,1:有害',
    pred_label INT DEFAULT -1,
    pred_prob FLOAT DEFAULT 0.0,
    model_version VARCHAR(20),
    inference_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ab_experiment_log (
    post_id VARCHAR(64) PRIMARY KEY,
    model_group VARCHAR(10) NOT NULL COMMENT 'control or experiment',
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
