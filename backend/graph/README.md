# Legacy Graph Boundary

`backend/graph/` is the rollback and Shadow comparison baseline for the V1.6 migration.

Production API traffic enters `backend.agent.runtime.run_rag_runtime`; it must not import this package directly. Keep this directory while `AGENT_RUNTIME_MODE=legacy` or `react_shadow` is in use. After the ReAct runtime passes offline, Shadow, staged traffic, permission, citation, latency and cost gates, remove the legacy nodes in a dedicated cleanup change.

Do not add new product behavior here. New orchestration belongs in `backend/agent/`; reusable retrieval, memory and generation primitives belong in `backend/services/`.
