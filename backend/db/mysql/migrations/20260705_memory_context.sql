-- Memory and context engineering schema migration.
-- MySQL 8.0+. The application also performs compatible create_all/schema sync.

ALTER TABLE `sessions`
    ADD COLUMN IF NOT EXISTS `title` VARCHAR(100) NULL;

UPDATE `sessions`
SET `title` = LEFT(`summary`, 100)
WHERE (`title` IS NULL OR `title` = '')
  AND `summary` IS NOT NULL;

ALTER TABLE `messages`
    ADD COLUMN IF NOT EXISTS `request_id` VARCHAR(64) NULL;

CREATE UNIQUE INDEX `uq_messages_request_role`
    ON `messages` (`request_id`, `role`);

CREATE TABLE IF NOT EXISTS `session_memories` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `session_id` VARCHAR(64) NOT NULL,
    `summary` JSON NULL,
    `summary_text` TEXT NULL,
    `summary_through_message_id` INT NULL,
    `version` INT NOT NULL DEFAULT 1,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_session_memories_session_id` (`session_id`),
    CONSTRAINT `fk_session_memories_session`
        FOREIGN KEY (`session_id`) REFERENCES `sessions` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `long_term_memories` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `memory_id` VARCHAR(64) NOT NULL,
    `user_id` INT NOT NULL,
    `kb_id` INT NULL,
    `scope_type` VARCHAR(20) NOT NULL DEFAULT 'user',
    `memory_type` VARCHAR(40) NOT NULL,
    `content` TEXT NOT NULL,
    `normalized_key` VARCHAR(255) NOT NULL,
    `keywords` JSON NULL,
    `confidence` DOUBLE NOT NULL DEFAULT 0.8,
    `status` VARCHAR(20) NOT NULL DEFAULT 'active',
    `source_session_id` VARCHAR(64) NULL,
    `source_message_id` INT NULL,
    `superseded_by` VARCHAR(64) NULL,
    `access_count` INT NOT NULL DEFAULT 0,
    `last_accessed_at` DATETIME NULL,
    `valid_from` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `valid_to` DATETIME NULL,
    `memory_metadata` JSON NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_long_term_memories_memory_id` (`memory_id`),
    KEY `ix_long_term_memory_scope`
        (`user_id`, `kb_id`, `scope_type`, `status`),
    KEY `ix_long_term_memory_lookup`
        (`user_id`, `memory_type`, `normalized_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `memory_update_logs` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `request_id` VARCHAR(64) NOT NULL,
    `user_id` INT NOT NULL,
    `session_id` VARCHAR(64) NULL,
    `memory_id` VARCHAR(64) NULL,
    `action` VARCHAR(30) NOT NULL,
    `before_value` JSON NULL,
    `after_value` JSON NULL,
    `reason` TEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `ix_memory_update_logs_request_id` (`request_id`),
    KEY `ix_memory_update_logs_user_id` (`user_id`),
    KEY `ix_memory_update_logs_session_id` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
