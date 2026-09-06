import { apiClient } from "./api"
import { type User, type UserProfile, type SessionContext, type SessionBranch } from "@/types"

export const userService = {
  /**
   * 获取当前用户信息
   */
  async getCurrentUser(): Promise<User> {
    return apiClient.get<User>("/users/me")
  },

  /**
   * 获取用户画像
   */
  async getProfile(): Promise<UserProfile> {
    return apiClient.get<UserProfile>("/users/me/profile")
  },

  /**
   * 更新用户画像
   */
  async updateProfile(profile: Partial<UserProfile>): Promise<UserProfile> {
    return apiClient.put<UserProfile>("/users/me/profile", profile)
  },

  /**
   * 获取所有会话
   */
  async getSessions(): Promise<SessionContext[]> {
    return apiClient.get<SessionContext[]>("/users/sessions")
  },

  /**
   * 创建新会话
   */
  async createSession(): Promise<SessionContext> {
    return apiClient.post<SessionContext>("/users/sessions", {})
  },

  /**
   * 删除会话
   */
  async deleteSession(sessionId: string): Promise<{ message: string }> {
    return apiClient.delete<{ message: string }>(`/users/sessions/${sessionId}`)
  },

  /**
   * 获取会话消息
   */
  async getSessionMessages(sessionId: string): Promise<{ messages: any[] }> {
    return apiClient.get<{ messages: any[] }>(`/users/sessions/${sessionId}/messages`)
  },

  /**
   * 更新会话分类
   */
  async updateSessionCategory(sessionId: string, category: string): Promise<any> {
    // 空字符串需要编码为特殊值，后端会处理为清除分类
    const categoryParam = category === "" ? "__null__" : category
    // 使用 get 方法的参数格式来构建 URL
    const url = `/users/sessions/${sessionId}?category=${encodeURIComponent(categoryParam)}`
    return apiClient.patch<any>(url)
  },
  async setSessionArchived(sessionId: string, archived: boolean): Promise<SessionContext> {
    return apiClient.patch<SessionContext>(`/users/sessions/${sessionId}?archived=${archived}`)
  },

  /**
   * 获取会话分类列表
   */
  async getSessionCategories(): Promise<{ categories: string[]; counts: Record<string, number> }> {
    return apiClient.get<{ categories: string[]; counts: Record<string, number> }>("/users/session-categories")
  },

  /**
   * 重命名会话
   */
  async renameSession(sessionId: string, title: string): Promise<SessionContext> {
    return apiClient.patch<SessionContext>(`/users/sessions/${sessionId}/title`, { title })
  },

  /**
   * 导出会话记录
   */
  async exportSession(sessionId: string, format: "json" | "markdown" | "txt" = "json"): Promise<void> {
    const token = localStorage.getItem("access_token")
    const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
    const url = `${baseURL}/api/v1/users/sessions/${sessionId}/export?format=${format}`

    // 使用 fetch 直接下载文件
    const response = await fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })

    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`)
    }

    // 获取文件名
    const contentDisposition = response.headers.get("Content-Disposition")
    let filename = `session_${sessionId}.${format === "markdown" ? "md" : format}`
    if (contentDisposition) {
      const match = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)
      if (match && match[1]) {
        filename = match[1].replace(/['"]/g, "")
      }
    }

    // 下载文件
    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = downloadUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  },

  async listBranches(sessionId: string): Promise<{ branches: SessionBranch[] }> {
    return apiClient.get(`/users/sessions/${sessionId}/branches`)
  },
  async createBranch(sessionId: string, fromMessageId: number, branchName?: string): Promise<SessionBranch> {
    return apiClient.post(`/users/sessions/${sessionId}/branches?from_message_id=${fromMessageId}${branchName ? `&branch_name=${encodeURIComponent(branchName)}` : ""}`, {})
  },
  async switchBranch(sessionId: string, branchId: number): Promise<{ message: string }> {
    return apiClient.post(`/users/sessions/${sessionId}/branches/${branchId}/switch`, {})
  },
  async deleteBranch(sessionId: string, branchId: number): Promise<{ message: string }> {
    return apiClient.delete(`/users/sessions/${sessionId}/branches/${branchId}`)
  },

  /**
   * 登录（TODO: 实现真正的认证）
   */
  async login(username: string, _password: string): Promise<User> {
    // 暂时返回模拟数据
    return {
      id: 1,
      username,
      is_active: true,
      created_at: new Date().toISOString(),
    }
  },

  /**
   * 注册（TODO: 实现真正的注册）
   */
  async register(username: string, email: string, _password: string): Promise<User> {
    // 暂时返回模拟数据
    return {
      id: 1,
      username,
      email,
      is_active: true,
      created_at: new Date().toISOString(),
    }
  },
}
