import { apiClient } from "./api"
import type { Feedback, TraceRun } from "@/types"

export const traceService = {
  async getSessionTrace(sessionId: string, requestId?: string): Promise<{ session_id: string; runs: TraceRun[] }> {
    return apiClient.get(`/users/sessions/${sessionId}/trace`, requestId ? { request_id: requestId } : undefined)
  },
  async submitFeedback(payload: { request_id: string; rating: "positive" | "negative"; comment?: string }): Promise<Feedback> {
    return apiClient.post("/rag/feedback", payload)
  },
}
