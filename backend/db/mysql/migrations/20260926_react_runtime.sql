-- Controlled ReAct runtime, durable tool idempotency and memory validity.
-- Apply with MySQL 8.0+. The online runtime is ReAct-only.

ALTER TABLE `rag_runs` ADD COLUMN IF NOT EXISTS `runtime_mode` VARCHAR(30) NOT NULL DEFAULT 'react';
ALTER TABLE `rag_runs` ADD COLUMN IF NOT EXISTS `iteration_count` INT NOT NULL DEFAULT 0;
ALTER TABLE `rag_runs` ADD COLUMN IF NOT EXISTS `stop_reason` VARCHAR(60) NULL;
ALTER TABLE `rag_runs` ADD COLUMN IF NOT EXISTS `tool_call_count` INT NOT NULL DEFAULT 0;
ALTER TABLE `rag_runs` ADD COLUMN IF NOT EXISTS `budget_profile` VARCHAR(40) NULL;
ALTER TABLE `rag_runs` ADD COLUMN IF NOT EXISTS `shadow_metrics` JSON NULL;

ALTER TABLE `long_term_memories` ADD COLUMN IF NOT EXISTS `conflict_group` VARCHAR(64) NULL;
ALTER TABLE `long_term_memories` ADD COLUMN IF NOT EXISTS `stale_at` DATETIME NULL;
ALTER TABLE `long_term_memories` ADD COLUMN IF NOT EXISTS `deletion_tombstone` TINYINT(1) NOT NULL DEFAULT 0;
ALTER TABLE `long_term_memories` ADD COLUMN IF NOT EXISTS `embedding_ref` VARCHAR(255) NULL;

CREATE TABLE IF NOT EXISTS `agent_tool_calls` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `run_id` VARCHAR(64) NOT NULL,
  `request_id` VARCHAR(64) NOT NULL,
  `tool_call_id` VARCHAR(128) NOT NULL,
  `user_id` INT NOT NULL,
  `session_id` VARCHAR(64) NOT NULL,
  `branch_id` INT NULL,
  `kb_id` INT NULL,
  `tool_name` VARCHAR(80) NOT NULL,
  `args_hash` VARCHAR(64) NOT NULL,
  `arguments_json` JSON NULL,
  `status` VARCHAR(30) NOT NULL DEFAULT 'started',
  `retry_count` INT NOT NULL DEFAULT 0,
  `latency_ms` INT NULL,
  `error_code` VARCHAR(80) NULL,
  `observation_json` JSON NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `completed_at` DATETIME NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_agent_tool_call_scope` (`run_id`, `tool_call_id`, `user_id`, `session_id`),
  KEY `ix_agent_tool_calls_request` (`request_id`),
  KEY `ix_agent_tool_calls_session` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `agent_observations` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `observation_id` VARCHAR(64) NOT NULL,
  `run_id` VARCHAR(64) NOT NULL,
  `tool_call_id` VARCHAR(128) NOT NULL,
  `user_id` INT NOT NULL,
  `session_id` VARCHAR(64) NOT NULL,
  `tool_name` VARCHAR(80) NOT NULL,
  `status` VARCHAR(30) NOT NULL,
  `summary` TEXT NOT NULL,
  `evidence_ids` JSON NULL,
  `supported_slots` JSON NULL,
  `missing_slots` JSON NULL,
  `conflicts` JSON NULL,
  `observation_json` JSON NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_agent_observations_id` (`observation_id`),
  KEY `ix_agent_observations_run` (`run_id`),
  KEY `ix_agent_observations_session` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
