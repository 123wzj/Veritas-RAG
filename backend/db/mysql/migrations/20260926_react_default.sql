-- Make the controlled ReAct runtime the database default for new trace rows.
-- Historical rows keep their original runtime_mode for accurate auditing.

ALTER TABLE `rag_runs`
    MODIFY COLUMN `runtime_mode` VARCHAR(30) NOT NULL DEFAULT 'react';
