import { apiClient } from "./api"
import { type RAGQueryRequest, type StreamEvent, type ChatMessage } from "@/types"

export const ragService = {
  /**
   * 流式 RAG 查询
   */
  async queryStream(
    request: RAGQueryRequest,
    onMessage: (event: StreamEvent) => void,
    onError?: (error: Error) => void
  ): Promise<() => void> {
    return apiClient.stream(
      "/rag/query/stream",
      request,
      (event) => {
        // event.data 是完整的 StreamEvent 对象（包含 event 和 data）
        onMessage(event.data as StreamEvent)
      },
      onError
    )
  },

  /**
   * 创建用户消息
   */
  createUserMessage(content: string): ChatMessage {
    return {
      id: crypto.randomUUID(),
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    }
  },

  /**
   * 创建助手消息
   */
  createAssistantMessage(): ChatMessage {
    return {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
      citations: [],
    }
  },
}
