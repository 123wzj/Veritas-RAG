# Legacy Graph Boundary

`backend/graph/` is the rollback and Shadow comparison baseline for the V1.6 migration.

Production API traffic enters `backend.agent.runtime.run_rag_runtime` and always executes `backend.agent.graph.react_graph`; the online runtime no longer imports this package. This directory is retained only for historical evaluation and migration comparison. New API features and fixes must not depend on it, and it may be removed after the remaining evaluation scripts are migrated.

Do not add new product behavior here. New orchestration belongs in `backend/agent/`; reusable retrieval, memory and generation primitives belong in `backend/services/`.
