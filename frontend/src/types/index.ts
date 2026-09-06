// ========== 用户类型 ==========
export interface User {
  id: number
  username: string
  email?: string
  is_active: boolean
  created_at: string
  updated_at?: string
}

export interface UserProfile {
  user_id: number
  preferred_language: string
  interests: string[]
  interaction_style: "concise" | "detailed" | "friendly"
  frequently_asked_topics: string[]
}

export interface SessionContext {
  session_id: string
  user_id: number
  kb_id?: number
  created_at: string
  last_active: string
  message_count: number
  title?: string
  summary?: string
  category?: string
}

// ========== 知识库类型 ==========
export type DocumentStatus = "pending" | "processing" | "completed" | "failed"
export type ModalityType = "text" | "image" | "table" | "code" | "mixed"

export interface KnowledgeBase {
  id: number
  user_id: number
  name: string
  description?: string
  acl_tags?: string[]
  document_count: number
  created_at: string
  updated_at?: string
}

export interface Document {
  id: number
  kb_id: number
  doc_id: string
  filename: string
  file_path: string
  file_size: number
  status: DocumentStatus
  modality: ModalityType
  language: string
  total_chunks: number
  error_message?: string
  created_at: string
  updated_at?: string
}

// ========== RAG 类型 ==========
export interface RAGQueryRequest {
  query: string
  session_id?: string
  kb_id?: number
  web_enabled: boolean
  stream: boolean
  top_k?: number
}

export interface Citation {
  source_type: "knowledge_base" | "web"
  doc_id: string
  chunk_id?: string
  parent_id?: string
  evidence_id?: string
  title: string
  page_no?: number
  section_path?: string
  url?: string
  snippet: string
  score?: number
  modality?: ModalityType
  image_url?: string
  caption?: string
  code_language?: string
}

export interface RAGQueryResponse {
  request_id: string
  answer: string
  confidence: number
  citations: Citation[]
  web_citations: Citation[]
  reasoning_summary: string
  used_web_enhancement: boolean
  reflection_notes?: string
  latency_ms: number
}

// ========== SSE 事件类型 ==========
export type StreamEventType =
  | "run.started"
  | "memory.loaded"
  | "query.rewritten"
  | "query.decomposed"
  | "route.planned"
  | "retrieval.started"
  | "retrieval.completed"
  | "retrieval.failed"
  | "rerank.completed"
  | "reflection.started"
  | "reflection.completed"
  | "websearch.started"
  | "websearch.completed"
  | "answer.delta"
  | "citation.delta"
  | "answer.completed"
  | "memory.updated"
  | "run.failed"

export interface StreamEvent {
  event: StreamEventType
  request_id?: string  // 后端可能不发送
  session_id?: string
  timestamp?: number  // 后端可能不发送
  data?: Record<string, unknown>
}

// ========== 聊天消息类型 ==========
export type MessageRole = "user" | "assistant" | "system"

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  timestamp: string
  citations?: Citation[]
  thinkingEvents?: StreamEvent[]
  thinkingCollapsed?: boolean
  metadata?: {
    confidence?: number
    reasoning_summary?: string
    used_web_enhancement?: boolean
  }
}

export interface ChatSession {
  session_id: string
  title: string
  created_at: string
  last_active: string
  message_count: number
  messages: ChatMessage[]
  category?: string
}
