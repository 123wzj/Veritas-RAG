import { useMemo, useState, type HTMLAttributes, type ReactNode } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { ChevronDown, ChevronRight, ExternalLink, ThumbsDown, ThumbsUp } from "lucide-react"

import { cn } from "@/lib/utils"
import { type ChatMessage } from "@/types"
import { traceService } from "@/services/trace"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"

interface ChatMessageProps {
  message: ChatMessage
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"
  const [manualThinkingCollapsed, setManualThinkingCollapsed] = useState<boolean | null>(null)
  const [referencesCollapsed, setReferencesCollapsed] = useState(true)
  const [expandedCitationIds, setExpandedCitationIds] = useState<Set<string>>(new Set())
  const [feedback, setFeedback] = useState<"positive" | "negative" | null>(null)
  const [comment, setComment] = useState("")
  const [feedbackNotice, setFeedbackNotice] = useState("")
  const [runCollapsed, setRunCollapsed] = useState(true)
  const [memoryCollapsed, setMemoryCollapsed] = useState(true)
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

  const toggleCitation = (key: string) => {
    setExpandedCitationIds((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

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
                  {citations.map((citation: any, index: number) => {
                    const citationKey = citation.evidence_id || citation.chunk_id || `${citation.doc_id}-${index}`
                    const expanded = expandedCitationIds.has(citationKey)
                    return (
                      <div key={citationKey} className="rounded-xl border border-border bg-white px-3 py-3 text-xs">
                        <button
                          type="button"
                          className="flex w-full items-center gap-2 text-left"
                          onClick={() => toggleCitation(citationKey)}
                        >
                          <span className="rounded-full bg-primary/10 px-2 py-0.5 font-semibold text-primary">
                            {citation.evidence_id || `E${index + 1}`}
                          </span>
                          <span className="font-medium text-foreground">{citation.title || citation.url || "未命名来源"}</span>
                          {citation.source_type === "web" && citation.url && (
                            <a
                              href={citation.url}
                              target="_blank"
                              rel="noreferrer"
                              className="inline-flex items-center text-primary hover:underline"
                              onClick={(event) => event.stopPropagation()}
                            >
                              打开 <ExternalLink className="ml-1 h-3 w-3" />
                            </a>
                          )}
                          {citation.section_path && <span className="text-muted-foreground">/{citation.section_path}</span>}
                          {citation.page_no && <span className="text-muted-foreground">第 {citation.page_no} 页</span>}
                          {typeof citation.score === "number" && <span className="text-muted-foreground">score {citation.score.toFixed(3)}</span>}
                          {citation.modality && <ModalityBadge modality={citation.modality} codeLanguage={citation.code_language} />}
                          <span className="ml-auto text-muted-foreground">
                            {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                          </span>
                        </button>
                        {expanded && renderCitationContent(citation)}
                      </div>
                    )
                  })}
                </div>
              </CollapsiblePanel>
            )}

            {message.metadata?.request_id && (
              <div className="mt-5 space-y-3">
                <div className="flex items-center gap-1">
                  <Button size="icon" variant={feedback === "positive" ? "secondary" : "ghost"} title="有帮助" onClick={() => setFeedback("positive")}><ThumbsUp className="h-4 w-4" /></Button>
                  <Button size="icon" variant={feedback === "negative" ? "secondary" : "ghost"} title="需要改进" onClick={() => setFeedback("negative")}><ThumbsDown className="h-4 w-4" /></Button>
                  {feedback && <Button size="sm" onClick={() => void traceService.submitFeedback({ request_id: message.metadata?.request_id || "", rating: feedback, comment: comment || undefined }).then(() => { setFeedbackNotice("反馈已保存") }).catch(() => { setFeedbackNotice("反馈提交失败") })}>提交反馈</Button>}
                  {feedbackNotice && <span role="status" className="text-xs text-muted-foreground">{feedbackNotice}</span>}
                </div>
                {feedback && <Textarea value={comment} onChange={(event) => setComment(event.target.value)} placeholder="可选：告诉我们哪里需要改进" rows={2} />}
                {message.metadata?.trace && <CollapsiblePanel title="运行详情" countLabel={`${message.metadata.trace.spans.length} 个 span`} collapsed={runCollapsed} onToggle={() => setRunCollapsed((value) => !value)}><div className="space-y-2 border-t border-border px-3 py-3 text-xs"><div>request_id: <code>{message.metadata.trace.request_id}</code></div><div>route_type: {message.metadata.trace.route_type || "未记录"} · answer_mode: {message.metadata.trace.answer_mode || "未记录"}</div><div>检索证据 {message.metadata.trace.selected_evidence_ids.length} 条 · 使用记忆 {message.metadata.trace.selected_memory_ids.length} 条 · 反思 {message.metadata.trace.reflection_count} 次</div>{message.metadata.trace.spans.map((span) => <div key={span.id} className="flex justify-between gap-3 rounded bg-white px-2 py-1"><span>{span.span_name}</span><span>{span.latency_ms ?? 0} ms</span></div>)}<div>最终状态：{message.metadata.trace.final_status}</div></div></CollapsiblePanel>}
                {message.metadata?.trace && message.metadata.trace.selected_memory_ids.length > 0 && <CollapsiblePanel title="使用的记忆" countLabel={`${message.metadata.trace.selected_memory_ids.length} 条`} collapsed={memoryCollapsed} onToggle={() => setMemoryCollapsed((value) => !value)}><div className="border-t border-border px-3 py-3 text-xs text-muted-foreground">{message.metadata.trace.selected_memory_ids.join("、")}</div></CollapsiblePanel>}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

const MODALITY_LABELS: Record<string, string> = {
  text: "文本",
  image: "图片",
  table: "表格",
  code: "代码",
  mixed: "混合",
}

function ModalityBadge({ modality, codeLanguage }: { modality: string; codeLanguage?: string }) {
  const label = modality === "code" ? (codeLanguage || "代码") : (MODALITY_LABELS[modality] || modality)
  const title = modality === "code" && codeLanguage ? `代码语言：${codeLanguage}` : undefined
  return (
    <span title={title} className="rounded-full border border-primary/20 bg-primary/5 px-2 py-0.5 font-medium text-primary">
      {label}
    </span>
  )
}

function renderCitationContent(citation: any) {
  // 图片证据：优先展示缩略图
  if (citation.modality === "image" && citation.image_url) {
    return (
      <div className="mt-2">
        <img
          src={citation.image_url}
          alt={citation.caption || citation.title || "图片证据"}
          className="max-h-48 w-auto rounded-lg border border-border object-contain"
        />
        {citation.caption && <p className="mt-1 text-muted-foreground">{citation.caption}</p>}
      </div>
    )
  }
  // 表格/代码证据：整块渲染 Markdown（GFM 支持表格与代码围栏）
  if (citation.modality === "table" || citation.modality === "code") {
    return (
      <div className="mt-2 max-h-56 overflow-auto rounded-lg border border-border bg-muted/40 p-2">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{citation.snippet || citation.title || ""}</ReactMarkdown>
      </div>
    )
  }
  // 文本证据：默认摘要
  if (citation.snippet) {
    return (
      <p className="mt-2 max-h-12 overflow-hidden leading-5 text-muted-foreground">{citation.snippet}</p>
    )
  }
  return null
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
