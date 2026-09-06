-- V1.5 governance, feedback and trace persistence. Apply with MySQL 8.0+.
ALTER TABLE `sessions` ADD COLUMN IF NOT EXISTS `archived` TINYINT(1) NOT NULL DEFAULT 0;
ALTER TABLE `long_term_memories` ADD COLUMN IF NOT EXISTS `source` VARCHAR(30) NOT NULL DEFAULT 'inferred';
ALTER TABLE `long_term_memories` ADD COLUMN IF NOT EXISTS `last_confirmed_at` DATETIME NULL;
ALTER TABLE `long_term_memories` ADD COLUMN IF NOT EXISTS `expires_at` DATETIME NULL;
CREATE TABLE IF NOT EXISTS `answer_feedback` (
  `id` INT NOT NULL AUTO_INCREMENT, `request_id` VARCHAR(64) NOT NULL, `session_id` VARCHAR(64) NULL,
  `user_id` INT NOT NULL, `rating` VARCHAR(20) NOT NULL, `comment` TEXT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`), UNIQUE KEY `uq_answer_feedback_request_user` (`request_id`, `user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `rag_runs` (
  `id` INT NOT NULL AUTO_INCREMENT, `request_id` VARCHAR(64) NOT NULL, `user_id` INT NOT NULL,
  `session_id` VARCHAR(64) NOT NULL, `kb_id` INT NULL, `route_type` VARCHAR(40) NULL,
  `final_status` VARCHAR(30) NOT NULL DEFAULT 'running', `answer_mode` VARCHAR(40) NULL,
  `reflection_count` INT NOT NULL DEFAULT 0, `total_latency_ms` INT NULL,
  `input_tokens` INT NOT NULL DEFAULT 0, `output_tokens` INT NOT NULL DEFAULT 0,
  `selected_evidence_ids` JSON NULL, `selected_memory_ids` JSON NULL, `error` TEXT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, `completed_at` DATETIME NULL,
  PRIMARY KEY (`id`), UNIQUE KEY `uq_rag_runs_request_id` (`request_id`), KEY `ix_rag_runs_session` (`session_id`), KEY `ix_rag_runs_user` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `rag_spans` (
  `id` INT NOT NULL AUTO_INCREMENT, `request_id` VARCHAR(64) NOT NULL, `span_name` VARCHAR(60) NOT NULL,
  `status` VARCHAR(20) NOT NULL DEFAULT 'completed', `started_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `ended_at` DATETIME NULL, `latency_ms` INT NULL, `model_name` VARCHAR(100) NULL,
  `input_tokens` INT NOT NULL DEFAULT 0, `output_tokens` INT NOT NULL DEFAULT 0,
  `metadata` JSON NULL, `error` TEXT NULL, PRIMARY KEY (`id`), KEY `ix_rag_spans_request` (`request_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
