import axios, { type AxiosInstance } from "axios"

// API 基础配置
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/api/v1`,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    })

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        // 添加认证 token
        const token = localStorage.getItem("access_token")
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => Promise.reject(error)
    )

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        // 暂时禁用登录跳转，允许无认证访问
        // if (error.response?.status === 401) {
        //   localStorage.removeItem("access_token")
        //   window.location.href = "/login"
        // }
        return Promise.reject(error)
      }
    )
  }

  // GET 请求
  async get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
    const response = await this.client.get<T>(url, { params })
    return response.data
  }

  // POST 请求
  async post<T>(url: string, data?: unknown): Promise<T> {
    const response = await this.client.post<T>(url, data)
    return response.data
  }

  // PUT 请求
  async put<T>(url: string, data?: unknown): Promise<T> {
    const response = await this.client.put<T>(url, data)
    return response.data
  }

  // PATCH 请求
  async patch<T>(url: string, data?: unknown, params?: Record<string, unknown>): Promise<T> {
    const config: any = {}
    if (params) {
      config.params = params
    }
    // 只有当 data 不是 null/undefined 时才发送 body
    if (data !== null && data !== undefined) {
      const response = await this.client.patch<T>(url, data, config)
      return response.data
    } else {
      const response = await this.client.patch<T>(url, config)
      return response.data
    }
  }

  // DELETE 请求
  async delete<T>(url: string): Promise<T> {
    const response = await this.client.delete<T>(url)
    return response.data
  }

  // SSE 流式请求
  async stream(
    url: string,
    data: unknown,
    onMessage: (event: MessageEvent) => void,
    onError?: (error: Error) => void,
    onComplete?: () => void
  ): Promise<() => void> {
    const token = localStorage.getItem("access_token")

    const response = await fetch(`${API_BASE_URL}/api/v1${url}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    let cancelled = false

    const read = async () => {
      try {
        while (!cancelled) {
          const { done, value } = await reader!.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split("\n")

          let currentEvent = ""

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim()
            } else if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6))
                // 构造符合 StreamEvent 类型的结构
                onMessage({ data: { event: currentEvent, data } } as MessageEvent)
              } catch (e) {
                console.error("Failed to parse SSE data:", e)
              }
            }
          }
        }
        onComplete?.()
      } catch (error) {
        if (!cancelled) {
          onError?.(error as Error)
        }
      }
    }

    read()

    // 返回取消函数
    return () => {
      cancelled = true
      reader?.cancel()
    }
  }
}

// 导出单例
export const apiClient = new ApiClient()
