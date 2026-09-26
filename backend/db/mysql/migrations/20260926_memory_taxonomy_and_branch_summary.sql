-- Three-category long-term memory taxonomy and branch-local short-term memory.
-- Column additions are idempotent on MySQL 8+; the index is guarded below.

ALTER TABLE `long_term_memories`
    ADD COLUMN IF NOT EXISTS `memory_category` VARCHAR(20) NOT NULL DEFAULT 'semantic' AFTER `scope_type`,
    ADD COLUMN IF NOT EXISTS `memory_payload` JSON NULL AFTER `content`;

SET @memory_category_index_exists := (
    SELECT COUNT(1)
    FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'long_term_memories'
      AND index_name = 'ix_long_term_memory_category'
);
SET @memory_category_index_sql := IF(
    @memory_category_index_exists = 0,
    'CREATE INDEX `ix_long_term_memory_category` ON `long_term_memories` (`user_id`, `memory_category`, `status`)',
    'SELECT 1'
);
PREPARE memory_category_index_stmt FROM @memory_category_index_sql;
EXECUTE memory_category_index_stmt;
DEALLOCATE PREPARE memory_category_index_stmt;

ALTER TABLE `conversation_branches`
    ADD COLUMN IF NOT EXISTS `summary` JSON NULL AFTER `parent_message_id`,
    ADD COLUMN IF NOT EXISTS `summary_text` TEXT NULL AFTER `summary`,
    ADD COLUMN IF NOT EXISTS `summary_through_message_id` INT NULL AFTER `summary_text`,
    ADD COLUMN IF NOT EXISTS `memory_version` INT NOT NULL DEFAULT 1 AFTER `summary_through_message_id`;

-- Existing records are stable facts/preferences in the old model.  The
-- business subtype lets us derive the safer cognitive category deterministically.
UPDATE `long_term_memories`
SET `memory_category` = CASE
    WHEN `memory_type` IN ('user_preference', 'project_constraint', 'interaction_rule', 'workflow')
        THEN 'procedural'
    WHEN `memory_type` IN ('episodic_event', 'interaction_episode')
        THEN 'episodic'
    ELSE 'semantic'
END
WHERE `memory_category` IS NULL OR `memory_category` = '' OR `memory_category` = 'semantic';
