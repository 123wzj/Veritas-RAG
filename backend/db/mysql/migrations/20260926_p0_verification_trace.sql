-- P0 verification and resumable trace hardening. MySQL 8.0+.

UPDATE `long_term_memories` SET `source` = 'explicit_user' WHERE `source` IN ('explicit', 'user');
UPDATE `long_term_memories` SET `source` = 'user_confirmed' WHERE `source` = 'confirmed';

ALTER TABLE `rag_spans`
  ADD COLUMN IF NOT EXISTS `attempt_no` INT NOT NULL DEFAULT 1 AFTER `request_id`,
  ADD COLUMN IF NOT EXISTS `node_name` VARCHAR(60) NULL AFTER `span_name`,
  ADD COLUMN IF NOT EXISTS `span_kind` VARCHAR(30) NOT NULL DEFAULT 'agent_node' AFTER `node_name`;

DELETE newer
FROM `rag_spans` AS newer
JOIN `rag_spans` AS older
  ON newer.`request_id` = older.`request_id`
 AND newer.`attempt_no` = older.`attempt_no`
 AND newer.`span_name` = older.`span_name`
 AND newer.`id` > older.`id`;

SET @span_unique_exists := (
  SELECT COUNT(*) FROM information_schema.statistics
  WHERE table_schema = DATABASE()
    AND table_name = 'rag_spans'
    AND index_name = 'uq_rag_span_attempt_name'
);
SET @span_unique_sql := IF(
  @span_unique_exists = 0,
  'CREATE UNIQUE INDEX `uq_rag_span_attempt_name` ON `rag_spans` (`request_id`, `attempt_no`, `span_name`)',
  'SELECT 1'
);
PREPARE span_unique_stmt FROM @span_unique_sql;
EXECUTE span_unique_stmt;
DEALLOCATE PREPARE span_unique_stmt;

CREATE TABLE IF NOT EXISTS `rag_run_attempts` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `request_id` VARCHAR(64) NOT NULL,
  `attempt_no` INT NOT NULL,
  `status` VARCHAR(30) NOT NULL DEFAULT 'running',
  `resumed` TINYINT(1) NOT NULL DEFAULT 0,
  `stop_reason` VARCHAR(60) NULL,
  `error` TEXT NULL,
  `started_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `ended_at` DATETIME NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_rag_run_attempt` (`request_id`, `attempt_no`),
  KEY `ix_rag_run_attempt_request` (`request_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `rag_events` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `request_id` VARCHAR(64) NOT NULL,
  `attempt_no` INT NOT NULL DEFAULT 1,
  `sequence_no` INT NOT NULL,
  `event_name` VARCHAR(80) NOT NULL,
  `node_name` VARCHAR(60) NULL,
  `status` VARCHAR(30) NULL,
  `metadata` JSON NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_rag_event_sequence` (`request_id`, `attempt_no`, `sequence_no`),
  KEY `ix_rag_events_request` (`request_id`),
  KEY `ix_rag_events_name` (`event_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `context_manifests` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `request_id` VARCHAR(64) NOT NULL,
  `attempt_no` INT NOT NULL DEFAULT 1,
  `node_name` VARCHAR(60) NOT NULL,
  `iteration` INT NOT NULL DEFAULT 0,
  `model_name` VARCHAR(100) NULL,
  `token_budget` INT NOT NULL DEFAULT 0,
  `token_used` INT NOT NULL DEFAULT 0,
  `loaded_sections` JSON NULL,
  `omitted_sections` JSON NULL,
  `section_usage` JSON NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_context_manifest_node_iteration` (`request_id`, `attempt_no`, `node_name`, `iteration`),
  KEY `ix_context_manifests_request` (`request_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
