import { useMemo, useState, type HTMLAttributes, type ReactNode } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { ChevronDown, ChevronRight, ExternalLink } from "lucide-react"

import { cn } from "@/lib/utils"
import { type ChatMessage } from "@/types"

interface ChatMessageProps {
  message: ChatMessage
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"
  const [manualThinkingCollapsed, setManualThinkingCollapsed] = useState<boolean | null>(null)
  const [referencesCollapsed, setReferencesCollapsed] = useState(true)
  const thinkingEvents = message.thinkingEvents || []
  const thinkingCollapsed = manualThinkingCollapsed ?? message.thinkingCollapsed ?? false

  const citations = useMemo(() => {
    const seen = new Set<string>()
    return (message.citations || []).filter((citation: any, index: number) => {
      const key = citation.evidence_id || citation.chunk_id || citation.parent_id || `${citation.doc_id}-${index}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  }, [message.citations])

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "px-5 py-4",
          isUser
            ? "max-w-[min(78%,42rem)] rounded-3xl bg-muted text-foreground"
            : "w-full max-w-none rounded-none bg-transparent"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-sm leading-7">{message.content}</p>
        ) : (
          <>
            {thinkingEvents.length > 0 && (
              <CollapsiblePanel
                title="处理过程"
                countLabel={`${thinkingEvents.length} 步`}
                collapsed={thinkingCollapsed}
                onToggle={() => setManualThinkingCollapsed(!thinkingCollapsed)}
                className="mb-4"
              >
                <div className="space-y-2 border-t border-border px-3 py-3">
                  {thinkingEvents.map((event, index) => (
                    <div key={`${event.event}-${index}`} className="rounded-xl bg-muted/70 px-3 py-2 text-xs">
                      <span className="font-medium text-foreground">{formatThinkingEvent(event.event)}</span>
                      <span className="ml-2 text-muted-foreground">{formatThinkingData(event.data)}</span>
                    </div>
                  ))}
                </div>
              </CollapsiblePanel>
            )}

            <div className="prose prose-sm max-w-none prose-p:leading-7">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  p: ({ children }: HTMLAttributes<HTMLParagraphElement>) => <p className="mb-3 last:mb-0">{children}</p>,
                  ul: ({ children }: HTMLAttributes<HTMLUListElement>) => <ul className="mb-3 list-disc pl-5">{children}</ul>,
                  ol: ({ children }: HTMLAttributes<HTMLOListElement>) => <ol className="mb-3 list-decimal pl-5">{children}</ol>,
                  li: ({ children }: HTMLAttributes<HTMLLIElement>) => <li className="mb-1.5">{children}</li>,
                  code: ({ className, children, ...props }: HTMLAttributes<HTMLElement>) => {
                    const isInline = !className?.includes("language-")
                    return isInline ? (
                      <code className="rounded-md bg-primary/10 px-1.5 py-0.5 text-primary" {...props}>
                        {children}
                      </code>
                    ) : (
                      <code className="block overflow-x-auto rounded-xl bg-slate-950/5 p-4 text-sm" {...props}>
                        {children}
                      </code>
                    )
                  },
                }}
              >
                {message.content || "正在组织答案..."}
              </ReactMarkdown>
            </div>

            {citations.length > 0 && (
              <CollapsiblePanel
                title="参考来源"
                countLabel={`${citations.length} 条`}
                collapsed={referencesCollapsed}
                onToggle={() => setReferencesCollapsed((value) => !value)}
                className="mt-5"
              >
                <div className="space-y-2.5 border-t border-border px-3 py-3">
                  {citations.map((citation: any, index: number) => (
                    <div key={citation.evidence_id || citation.chunk_id || index} className="rounded-xl border border-border bg-white px-3 py-3 text-xs">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-full bg-primary/10 px-2 py-0.5 font-semibold text-primary">
                          {citation.evidence_id || `E${index + 1}`}
                        </span>
                        <span className="font-medium text-foreground">{citation.title || citation.url || "未命名来源"}</span>
                        {citation.source_type === "web" && citation.url && (
                          <a href={citation.url} target="_blank" rel="noreferrer" className="inline-flex items-center text-primary hover:underline">
                            打开 <ExternalLink className="ml-1 h-3 w-3" />
                          </a>
                        )}
                        {citation.section_path && <span className="text-muted-foreground">/{citation.section_path}</span>}
                        {citation.page_no && <span className="text-muted-foreground">第 {citation.page_no} 页</span>}
                        {typeof citation.score === "number" && <span className="text-muted-foreground">score {citation.score.toFixed(3)}</span>}
                      </div>
                      {citation.snippet && (
                        <p className="mt-2 max-h-12 overflow-hidden leading-5 text-muted-foreground">{citation.snippet}</p>
                      )}
                    </div>
                  ))}
                </div>
              </CollapsiblePanel>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function CollapsiblePanel({
  title,
  countLabel,
  collapsed,
  onToggle,
  className,
  children,
}: {
  title: string
  countLabel?: string
  collapsed: boolean
  onToggle: () => void
  className?: string
  children: ReactNode
}) {
  return (
    <div className={cn("rounded-xl border border-border bg-muted/45", className)}>
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
        onClick={onToggle}
      >
        {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        <span>{title}</span>
        {countLabel && <span className="ml-auto">{countLabel}</span>}
      </button>
      {!collapsed && children}
    </div>
  )
}

function formatThinkingEvent(event: string) {
  const labels: Record<string, string> = {
    "run.started": "开始处理",
    "memory.loaded": "加载记忆",
    "query.rewritten": "问题改写",
    "query.decomposed": "查询拆解",
    "route.planned": "路由规划",
    "retrieval.started": "知识库检索",
    "retrieval.completed": "检索完成",
    "rerank.completed": "精排完成",
    "reflection.started": "反思校验",
    "reflection.completed": "反思完成",
    "websearch.started": "联网搜索",
    "websearch.completed": "联网完成",
    "answer.completed": "回答完成",
    "run.failed": "处理失败",
  }
  return labels[event] || event
}

function formatThinkingData(data?: Record<string, unknown>) {
  if (!data) return ""
  if (typeof data.count === "number") return `命中 ${data.count} 条`
  if (typeof data.sub_query_count === "number") return `${data.sub_query_count} 个子问题`
  if (Array.isArray(data.queries)) return data.queries.slice(0, 2).join(" / ")
  if (typeof data.route_type === "string") return `路由：${data.route_type}`
  if (typeof data.citations_count === "number") return `引用 ${data.citations_count} 条`
  if (typeof data.error === "string") return data.error
  return ""
}
