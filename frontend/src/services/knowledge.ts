import { apiClient } from "./api"
import { type KnowledgeBase, type Document, type Capabilities } from "@/types"

export const knowledgeService = {
  /**
   * 获取知识库列表
   */
  async getKnowledgeBases(): Promise<KnowledgeBase[]> {
    return apiClient.get<KnowledgeBase[]>("/knowledge/")
  },

  /**
   * 创建知识库
   */
  async createKnowledgeBase(
    data: Pick<KnowledgeBase, "name" | "description" | "acl_tags">
  ): Promise<KnowledgeBase> {
    return apiClient.post<KnowledgeBase>("/knowledge/", data)
  },

  /**
   * 获取知识库详情
   */
  async getKnowledgeBase(kbId: number): Promise<KnowledgeBase> {
    return apiClient.get<KnowledgeBase>(`/knowledge/${kbId}`)
  },

  /**
   * 更新知识库
   */
  async updateKnowledgeBase(
    kbId: number,
    data: Partial<Pick<KnowledgeBase, "name" | "description" | "acl_tags">>
  ): Promise<KnowledgeBase> {
    return apiClient.put<KnowledgeBase>(`/knowledge/${kbId}`, data)
  },

  /**
   * 删除知识库
   */
  async deleteKnowledgeBase(kbId: number): Promise<{ message: string }> {
    return apiClient.delete<{ message: string }>(`/knowledge/${kbId}`)
  },

  /**
   * 上传文档
   */
  async uploadDocument(
    kbId: number,
    file: File,
    onProgress?: (progress: number) => void,
    options?: { confirmSameName?: boolean }
  ): Promise<{
    message: string
    doc_id: string
    filename: string
    kb_id: number
    status: string
    parent_count: number
    child_count: number
  }> {
    const formData = new FormData()
    formData.append("file", file)

    const token = localStorage.getItem("access_token")
    const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
    const confirmParam = options?.confirmSameName ? "?confirm_same_name=true" : ""

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      })

      xhr.addEventListener("load", () => {
        if (xhr.status === 200) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          let detail: any = null
          try {
            detail = JSON.parse(xhr.responseText)?.detail
          } catch {
            detail = null
          }
          const message = detail?.message || `Upload failed: ${xhr.statusText}`
          const error = new Error(message) as Error & { status?: number; detail?: any }
          error.status = xhr.status
          error.detail = detail
          reject(error)
        }
      })

      xhr.addEventListener("error", () => {
        reject(new Error("Upload failed"))
      })

      xhr.open("POST", `${baseURL}/api/v1/knowledge/${kbId}/upload${confirmParam}`)
      if (token) {
        xhr.setRequestHeader("Authorization", `Bearer ${token}`)
      }
      xhr.send(formData)
    })
  },

  /**
   * 获取文档列表
   */
  async getDocuments(kbId: number): Promise<Document[]> {
    return apiClient.get<Document[]>(`/knowledge/${kbId}/documents`)
  },
  async getCapabilities(kbId: number): Promise<Capabilities> {
    return apiClient.get<Capabilities>(`/knowledge/${kbId}/capabilities`)
  },

  /**
   * 获取文档详情
   */
  async getDocument(kbId: number, docId: string): Promise<Document> {
    return apiClient.get<Document>(`/knowledge/${kbId}/documents/${docId}`)
  },

  /**
   * 删除文档
   */
  async deleteDocument(kbId: number, docId: string): Promise<{ message: string }> {
    return apiClient.delete<{ message: string }>(`/knowledge/${kbId}/documents/${docId}`)
  },
}
