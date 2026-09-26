-- A branch is a lineage record pointing to a fully independent session.
-- Checkpoint data is stored in the configured LangGraph SQLite checkpoint DB.

ALTER TABLE `conversation_branches`
    ADD COLUMN IF NOT EXISTS `forked_session_id` VARCHAR(64) NULL AFTER `parent_message_id`;

SET @forked_session_index_exists := (
    SELECT COUNT(1)
    FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'conversation_branches'
      AND index_name = 'uq_conversation_branches_forked_session_id'
);
SET @forked_session_index_sql := IF(
    @forked_session_index_exists = 0,
    'CREATE UNIQUE INDEX `uq_conversation_branches_forked_session_id` ON `conversation_branches` (`forked_session_id`)',
    'SELECT 1'
);
PREPARE forked_session_index_stmt FROM @forked_session_index_sql;
EXECUTE forked_session_index_stmt;
DEALLOCATE PREPARE forked_session_index_stmt;

-- These columns were introduced by an unreleased branch-summary design and
-- are intentionally removed. Each fork now owns a normal session_memories row.
ALTER TABLE `conversation_branches`
    DROP COLUMN IF EXISTS `summary`,
    DROP COLUMN IF EXISTS `summary_text`,
    DROP COLUMN IF EXISTS `summary_through_message_id`,
    DROP COLUMN IF EXISTS `memory_version`;
