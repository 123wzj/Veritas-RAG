import { apiClient } from "./api"
import type { Memory, MemoryAuditEntry } from "@/types"

export const memoryService = {
  async list(params?: { status?: string; memory_type?: string; scope_type?: string; page?: number; page_size?: number }): Promise<{ items: Memory[]; page: number; page_size: number; total: number }> {
    return apiClient.get("/memory/long-term", params)
  },
  async update(memoryId: string, payload: { content?: string; status?: string; confidence?: number; expires_at?: string | null; operation?: string; request_id?: string }): Promise<Memory> {
    return apiClient.patch(`/memory/long-term/${memoryId}`, payload)
  },
  async remove(memoryId: string): Promise<{ message: string; memory: Memory }> {
    return apiClient.delete(`/memory/long-term/${memoryId}`)
  },
  async audit(memoryId?: string): Promise<{ items: MemoryAuditEntry[] }> {
    return apiClient.get("/memory/updates", memoryId ? { memory_id: memoryId } : undefined)
  },
}
