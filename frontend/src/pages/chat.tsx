import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react"
import {
  Archive,
  Database,
  Download,
  Globe2,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Search,
  Tag,
  Trash2,
} from "lucide-react"

import { ChatInput } from "@/components/chat/chat-input"
import { ChatMessage } from "@/components/chat/chat-message"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ragService } from "@/services/rag"
import { userService } from "@/services/user"
import { useChatStore } from "@/stores/chat"
import { useKnowledgeStore } from "@/stores/knowledge"
import { type ChatMessage as ChatMessageType, type ChatSession, type SessionContext } from "@/types"
import { cn } from "@/lib/utils"

const predefinedCategories = ["工作", "学习", "项目", "面试", "资料", "其他"]
const thinkingEvents = new Set([
  "run.started",
  "memory.loaded",
  "query.rewritten",
  "query.decomposed",
  "route.planned",
  "retrieval.started",
  "retrieval.completed",
  "rerank.completed",
  "reflection.started",
  "reflection.completed",
  "websearch.started",
  "websearch.completed",
])

const toChatSession = (session: SessionContext, messages: ChatMessageType[] = []): ChatSession => ({
  session_id: session.session_id,
  title: session.summary || "新对话",
  created_at: session.created_at,
  last_active: session.last_active,
  message_count: session.message_count,
  category: session.category || undefined,
  messages,
})

const toChatMessages = (messages: any[]): ChatMessageType[] =>
  messages.map((message) => ({
    id: String(message.id ?? crypto.randomUUID()),
    role: message.role,
    content: message.content,
    citations: message.citations || [],
    timestamp: message.created_at || new Date().toISOString(),
  }))

function formatSessionTime(value: string) {
  return new Date(value).toLocaleString([], {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function ChatPage() {
  const store = useChatStore()
  const { currentSession, sessions, isStreaming } = useChatStore()
  const currentKbId = useKnowledgeStore((state) => state.currentKbId)
  const knowledgeBases = useKnowledgeStore((state) => state.knowledgeBases)
  const didBootstrapSelection = useRef(false)

  const [webSearchEnabled, setWebSearchEnabled] = useState(false)
  const [knowledgeSearchEnabled, setKnowledgeSearchEnabled] = useState(true)
  const [categories, setCategories] = useState<string[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [isDraftMode, setIsDraftMode] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const currentKnowledgeBase = useMemo(
    () => knowledgeBases.find((item) => item.id === currentKbId),
    [knowledgeBases, currentKbId]
  )
  const visibleMessages = currentSession?.messages || []

  const mode = useMemo(() => {
    if (webSearchEnabled && knowledgeSearchEnabled) {
      return {
        label: "混合增强",
        desc: "先检索知识库，再补充联网搜索，适合既要内部资料又要时效信息的问题。",
      }
    }
    if (knowledgeSearchEnabled) {
      return { label: "知识库检索", desc: "仅基于当前知识库召回证据并生成带引用回答。" }
    }
    if (webSearchEnabled) {
      return { label: "联网搜索", desc: "使用外部搜索补充实时信息，不依赖知识库。" }
    }
    return { label: "模型直答", desc: "不启用外部增强，直接由模型回答。" }
  }, [knowledgeSearchEnabled, webSearchEnabled])

  const loadCategories = async () => {
    try {
      const data = await userService.getSessionCategories()
      setCategories(data.categories || [])
    } catch (error) {
      console.error("Failed to load categories:", error)
    }
  }

  const loadSessions = async () => {
    try {
      const sessionData = await userService.getSessions()
      const mappedSessions = sessionData.map((session) => toChatSession(session))
      store.setSessions(mappedSessions)
      if (!didBootstrapSelection.current) {
        didBootstrapSelection.current = true
        if (!currentSession && mappedSessions.length > 0) store.setCurrentSession(mappedSessions[0])
      }
    } catch (error) {
      console.error("Failed to load sessions:", error)
    }
  }

  const createPersistedSession = async (): Promise<ChatSession | null> => {
    try {
      const session = await userService.createSession()
      const chatSession = toChatSession(session)
      store.createSession(chatSession)
      return chatSession
    } catch (error) {
      console.error("Failed to create session:", error)
      return null
    }
  }

  useEffect(() => {
    void Promise.all([loadSessions(), loadCategories()])
  }, [])

  useEffect(() => {
    if (!didBootstrapSelection.current || isDraftMode) return
    if (!currentSession && sessions.length > 0) store.setCurrentSession(sessions[0])
  }, [currentSession, sessions, isDraftMode, store])

  const handleNewSession = () => {
    setIsDraftMode(true)
    store.setCurrentSession(null)
  }

  const handleSelectSession = async (sessionId: string) => {
    const session = sessions.find((item) => item.session_id === sessionId)
    if (!session) return
    setIsDraftMode(false)
    store.setCurrentSession(session)
    if (session.messages.length > 0) return
    try {
      const data = await userService.getSessionMessages(sessionId)
      store.setMessages(toChatMessages(data.messages))
    } catch (error) {
      console.error("Failed to load session messages:", error)
    }
  }

  const handleDeleteSession = async (sessionId: string, event: MouseEvent) => {
    event.stopPropagation()
    if (!window.confirm("确定删除这个会话吗？")) return
    const wasCurrent = currentSession?.session_id === sessionId
    try {
      await userService.deleteSession(sessionId)
      store.deleteSession(sessionId)
      if (wasCurrent) {
        const remaining = useChatStore.getState().sessions
        store.setCurrentSession(remaining[0] || null)
        setIsDraftMode(remaining.length === 0)
      }
      await Promise.all([loadSessions(), loadCategories()])
    } catch (error) {
      console.error("Failed to delete session:", error)
      window.alert("删除会话失败，请稍后重试。")
    }
  }

  const handleRenameSession = async (sessionId: string, event: MouseEvent) => {
    event.stopPropagation()
    const session = sessions.find((item) => item.session_id === sessionId)
    const currentTitle = session?.title || "新对话"
    const title = window.prompt("请输入新的会话名称", currentTitle)?.trim()
    if (!title || title === currentTitle) return
    try {
      const updated = await userService.renameSession(sessionId, title)
      store.upsertSession(toChatSession(updated, session?.messages || []))
      await loadSessions()
    } catch (error) {
      console.error("Failed to rename session:", error)
      window.alert("重命名失败，请稍后重试。")
    }
  }

  const handleExportSession = async (sessionId: string, format: "json" | "markdown" | "txt", event: MouseEvent) => {
    event.stopPropagation()
    try {
      await userService.exportSession(sessionId, format)
    } catch (error) {
      console.error("Failed to export session:", error)
      window.alert("导出失败，请稍后重试。")
    }
  }

  const handleSetCategory = async (sessionId: string, category: string, event: MouseEvent) => {
    event.stopPropagation()
    try {
      await userService.updateSessionCategory(sessionId, category)
      await Promise.all([loadSessions(), loadCategories()])
    } catch (error) {
      console.error("Failed to update category:", error)
      window.alert("设置分类失败，请稍后重试。")
    }
  }

  const handleSendMessage = async (content: string) => {
    if (isStreaming) return
    let workingSession = currentSession
    if (!workingSession) {
      workingSession = await createPersistedSession()
      if (!workingSession) {
        window.alert("创建会话失败，请稍后重试。")
        return
      }
    }

    setIsDraftMode(false)
    store.setCurrentSession(workingSession)
    store.addMessage(ragService.createUserMessage(content))
    store.addMessage(ragService.createAssistantMessage())
    store.setStreaming(true)

    let answerContent = ""
    let citations: any[] = []
    try {
      await ragService.queryStream(
        {
          query: content,
          session_id: workingSession.session_id,
          web_enabled: webSearchEnabled,
          kb_id: knowledgeSearchEnabled ? currentKbId ?? undefined : undefined,
          stream: true,
        },
        (event: any) => {
          if (thinkingEvents.has(event.event)) store.appendThinkingEventToLastMessage(event)
          switch (event.event) {
            case "answer.delta":
              answerContent += event.data?.text || ""
              store.updateLastMessage(answerContent, citations)
              break
            case "citation.delta":
              citations = [...citations, event.data]
              store.updateLastMessage(answerContent, citations)
              break
            case "answer.completed":
              store.updateLastMessage(event.data?.answer || answerContent, event.data?.citations || citations)
              store.setLastMessageThinkingCollapsed(true)
              store.setStreaming(false)
              void Promise.all([loadSessions(), loadCategories()])
              break
            case "run.failed":
              store.appendThinkingEventToLastMessage(event)
              store.setLastMessageThinkingCollapsed(false)
              store.updateLastMessage("抱歉，查询失败，请稍后重试。", [])
              store.setStreaming(false)
              break
            default:
              break
          }
        },
        (error) => {
          console.error("Query stream error:", error)
          store.updateLastMessage("抱歉，查询失败，请稍后重试。", [])
          store.setStreaming(false)
        }
      )
    } catch (error) {
      console.error("Query error:", error)
      store.updateLastMessage("抱歉，查询失败，请稍后重试。", [])
      store.setStreaming(false)
    }
  }

  const filteredSessions = selectedCategory
    ? sessions.filter((session) => session.category === selectedCategory)
    : sessions

  return (
    <div className="flex h-[calc(100vh-57px)] min-h-0 bg-white lg:h-screen">
      {sidebarOpen && (
        <aside className="hidden w-[300px] shrink-0 border-r border-border bg-white xl:flex xl:flex-col">
          <div className="flex h-14 items-center gap-2 border-b border-border px-3">
            <Button onClick={handleNewSession} className="flex-1 justify-start rounded-xl" variant="ghost">
              <MessageSquarePlus className="mr-2 h-4 w-4" />
              新建对话
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 rounded-xl"
              onClick={() => setSidebarOpen(false)}
              aria-label="收起会话栏"
            >
              <PanelLeftClose className="h-4 w-4" />
            </Button>
          </div>

          <div className="m-3 rounded-2xl border border-border bg-muted/45 p-3">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">会话分类</p>
            <div className="flex flex-wrap gap-2">
              <button
                className={cn("rounded-full px-3 py-1 text-xs", !selectedCategory ? "bg-primary text-primary-foreground" : "bg-white text-muted-foreground")}
                onClick={() => setSelectedCategory(null)}
              >
                全部
              </button>
              {[...predefinedCategories, ...categories.filter((item) => !predefinedCategories.includes(item))].map((category) => (
                <button
                  key={category}
                  className={cn("rounded-full px-3 py-1 text-xs", selectedCategory === category ? "bg-primary text-primary-foreground" : "bg-white text-muted-foreground")}
                  onClick={() => setSelectedCategory(category)}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>

          <ScrollArea className="min-h-0 flex-1 px-3 pb-3">
            <div className="space-y-1.5">
              {filteredSessions.map((session) => (
                <div
                  key={session.session_id}
                  onClick={() => void handleSelectSession(session.session_id)}
                  className={cn(
                    "group cursor-pointer rounded-xl p-3 transition-colors",
                    currentSession?.session_id === session.session_id ? "bg-muted text-foreground" : "hover:bg-muted/60"
                  )}
                >
                  <div className="flex items-start gap-3">
                    <Archive className="mt-1 h-4 w-4 shrink-0 text-primary" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{session.title}</p>
                      <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                        <span>{formatSessionTime(session.last_active)}</span>
                        <span>{session.message_count} 条消息</span>
                      </div>
                      {session.category && <span className="mt-2 inline-block rounded-full bg-secondary px-2 py-0.5 text-[11px]">{session.category}</span>}
                    </div>
                  </div>
                  <div className="mt-3 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(event) => void handleRenameSession(session.session_id, event)}>
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(event) => event.stopPropagation()}>
                          <Tag className="h-3.5 w-3.5" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuLabel>设置分类</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        {predefinedCategories.map((category) => (
                          <DropdownMenuItem key={category} onClick={(event) => void handleSetCategory(session.session_id, category, event)}>
                            {category}
                          </DropdownMenuItem>
                        ))}
                        <DropdownMenuSeparator />
                        <DropdownMenuItem onClick={(event) => void handleSetCategory(session.session_id, "", event)}>清除分类</DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(event) => event.stopPropagation()}>
                          <Download className="h-3.5 w-3.5" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={(event) => void handleExportSession(session.session_id, "json", event)}>JSON</DropdownMenuItem>
                        <DropdownMenuItem onClick={(event) => void handleExportSession(session.session_id, "markdown", event)}>Markdown</DropdownMenuItem>
                        <DropdownMenuItem onClick={(event) => void handleExportSession(session.session_id, "txt", event)}>TXT</DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                    <Button variant="ghost" size="icon" className="ml-auto h-8 w-8 text-destructive" onClick={(event) => void handleDeleteSession(session.session_id, event)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </aside>
      )}

      <section className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="border-b border-border bg-white px-4 py-2.5">
          <div className="flex flex-wrap items-center gap-3">
            {!sidebarOpen && (
              <Button
                variant="ghost"
                size="icon"
                className="hidden h-9 w-9 rounded-xl xl:inline-flex"
                onClick={() => setSidebarOpen(true)}
                aria-label="展开会话栏"
              >
                <PanelLeftOpen className="h-4 w-4" />
              </Button>
            )}
            <div className="min-w-0 flex-1">
              <h1 className="truncate text-sm font-medium">{currentSession?.title || "新对话"}</h1>
              <p className="mt-0.5 truncate text-xs text-muted-foreground">
                {mode.label} · {currentKnowledgeBase?.name || "未选择知识库"}
              </p>
            </div>
            <ModeSwitch
              icon={Database}
              label="知识库"
              active={knowledgeSearchEnabled}
              onClick={() => setKnowledgeSearchEnabled((value) => !value)}
            />
            <ModeSwitch
              icon={Globe2}
              label="联网"
              active={webSearchEnabled}
              onClick={() => setWebSearchEnabled((value) => !value)}
            />
            <Button variant="outline" className="rounded-xl xl:hidden" onClick={() => setSidebarOpen((value) => !value)}>
              历史
            </Button>
          </div>
          <div className="mt-2 grid gap-2 lg:grid-cols-[1fr_auto]">
            <div className="rounded-xl bg-muted/45 px-3 py-2">
              <div className="flex items-start gap-3">
                <Search className="mt-0.5 h-4 w-4 text-primary" />
                <div>
                  <p className="text-xs leading-5 text-muted-foreground">{mode.desc}</p>
                </div>
              </div>
            </div>
            <div className="rounded-xl bg-muted/45 px-3 py-2 text-sm">
              <p className="text-xs text-muted-foreground">{currentKnowledgeBase ? `${currentKnowledgeBase.document_count || 0} 个文档可检索` : "可在知识库页面选择数据源"}</p>
            </div>
          </div>
        </header>

        <div className="min-h-0 flex-1">
          <div className="mx-auto flex h-full max-w-4xl flex-col px-4">
            <ScrollArea className="min-h-0 flex-1 py-5">
              {visibleMessages.length ? (
                <div className="flex flex-col gap-5 pb-4">
                  {visibleMessages.map((message) => (
                    <ChatMessage key={message.id} message={message} />
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center">
                  <div className="max-w-2xl text-center">
                    <p className="text-xs font-semibold uppercase tracking-wider text-primary">Ready</p>
                    <h2 className="mt-3 text-3xl font-semibold tracking-tight">今天想查什么？</h2>
                    <p className="mt-4 text-sm leading-7 text-muted-foreground">
                      同时开启知识库和联网时，后端会先走知识库检索，再补充联网搜索，最终把两类证据一起用于生成与引用。
                    </p>
                  </div>
                </div>
              )}
            </ScrollArea>
            <div className="bg-white pb-4 pt-3">
              <ChatInput onSend={(message) => void handleSendMessage(message)} disabled={isStreaming} />
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

function ModeSwitch({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: typeof Database
  label: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-sm transition-colors",
        active ? "border-primary bg-primary text-primary-foreground" : "border-border bg-white text-muted-foreground hover:bg-muted"
      )}
    >
      <Icon className="h-4 w-4" />
      {label}
    </button>
  )
}
