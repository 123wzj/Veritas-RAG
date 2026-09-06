import { useEffect, useState } from "react"
import { Brain, Check, Pencil, RefreshCw, Trash2, X } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Select } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { memoryService } from "@/services/memory"
import type { Memory, MemoryAuditEntry } from "@/types"

const date = (value?: string | null) => value ? new Date(value).toLocaleString() : "未设置"

export function MemoryPage() {
  const [items, setItems] = useState<Memory[]>([])
  const [audit, setAudit] = useState<MemoryAuditEntry[]>([])
  const [status, setStatus] = useState("all")
  const [memoryType, setMemoryType] = useState("all")
  const [scopeType, setScopeType] = useState("all")
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState("")
  const [notice, setNotice] = useState<{ text: string; error?: boolean } | null>(null)
  const [loading, setLoading] = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      const result = await memoryService.list({ status: status === "all" ? undefined : status, memory_type: memoryType === "all" ? undefined : memoryType, scope_type: scopeType === "all" ? undefined : scopeType, page_size: 100 })
      setItems(result.items || [])
    } catch { setNotice({ text: "记忆加载失败，请稍后重试。", error: true }) } finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [status, memoryType, scopeType])

  const mutate = async (item: Memory, operation: "confirm" | "reject" | "delete") => {
    try { await memoryService.update(item.memory_id, { operation, request_id: crypto.randomUUID() }); setNotice({ text: operation === "confirm" ? "记忆已确认。" : "记忆状态已更新。" }); await load() }
    catch { setNotice({ text: "记忆更新失败，请稍后重试。", error: true }) }
  }
  const saveEdit = async (item: Memory) => {
    if (!draft.trim()) return
    try { await memoryService.update(item.memory_id, { content: draft.trim(), operation: "edit", request_id: crypto.randomUUID() }); setEditing(null); setNotice({ text: "记忆内容已保存。" }); await load() }
    catch { setNotice({ text: "保存失败，请稍后重试。", error: true }) }
  }
  const showAudit = async (memoryId: string) => {
    try { setAudit((await memoryService.audit(memoryId)).items || []) } catch { setNotice({ text: "审计记录加载失败。", error: true }) }
  }

  return <div className="min-h-screen p-5 lg:p-8"><div className="mx-auto max-w-6xl space-y-6">
    <header className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Memory</p><h1 className="mt-1 text-3xl font-semibold">长期记忆治理</h1><p className="mt-2 text-sm text-muted-foreground">推断记忆必须确认后才会作为事实使用。</p></div><Button variant="outline" size="icon" title="刷新" onClick={() => void load()}><RefreshCw className="h-4 w-4" /></Button></header>
    <div className="flex flex-wrap gap-2"><Select className="w-44" value={memoryType} onChange={(event) => setMemoryType(event.target.value)}><option value="all">全部类型</option><option value="profile">profile</option><option value="preference">preference</option><option value="constraint">constraint</option><option value="project_state">project_state</option></Select><Select className="w-36" value={scopeType} onChange={(event) => setScopeType(event.target.value)}><option value="all">全部作用域</option><option value="user">user</option><option value="project">project</option></Select><Select className="w-44" value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">全部状态</option><option value="active">active</option><option value="pending_confirmation">待确认</option><option value="rejected">rejected</option><option value="inactive">inactive</option></Select></div>
    {notice && <div role="status" className={notice.error ? "rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive" : "rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-sm"}>{notice.text}</div>}
    <div className="space-y-3">{loading ? <p className="text-sm text-muted-foreground">正在加载…</p> : items.length === 0 ? <p className="text-sm text-muted-foreground">暂无符合条件的记忆。</p> : items.map((item) => <div key={item.memory_id} className="rounded-xl border bg-white p-4"><div className="flex flex-wrap items-start justify-between gap-3"><div className="min-w-0 flex-1"><div className="flex flex-wrap gap-2"><Badge variant="secondary">{item.memory_type}</Badge><Badge variant="outline">{item.scope_type}</Badge><Badge variant={item.status === "pending_confirmation" ? "warning" : item.status === "active" ? "success" : "secondary"}>{item.status}</Badge><span className="text-xs text-muted-foreground">来源 {item.source} · 置信度 {Math.round(item.confidence * 100)}%</span></div>{editing === item.memory_id ? <div className="mt-3 space-y-2"><Textarea value={draft} onChange={(event) => setDraft(event.target.value)} /><div className="flex gap-2"><Button size="sm" onClick={() => void saveEdit(item)}><Check className="mr-1 h-3 w-3" />保存</Button><Button size="sm" variant="outline" onClick={() => setEditing(null)}><X className="mr-1 h-3 w-3" />取消</Button></div></div> : <p className="mt-2 text-sm leading-6">{item.content}</p>}<p className="mt-2 text-xs text-muted-foreground">创建 {date(item.created_at)} · 更新 {date(item.updated_at)} · 确认 {date(item.last_confirmed_at)} · 过期 {date(item.expires_at)}</p></div><div className="flex shrink-0 gap-1">{item.status === "pending_confirmation" && <><Button size="icon" variant="ghost" title="确认" onClick={() => void mutate(item, "confirm")}><Check className="h-4 w-4" /></Button><Button size="icon" variant="ghost" title="拒绝" onClick={() => void mutate(item, "reject")}><X className="h-4 w-4" /></Button></>}<Button size="icon" variant="ghost" title="编辑" onClick={() => { setEditing(item.memory_id); setDraft(item.content) }}><Pencil className="h-4 w-4" /></Button><Button size="icon" variant="ghost" title="审计记录" onClick={() => void showAudit(item.memory_id)}><Brain className="h-4 w-4" /></Button><Button size="icon" variant="ghost" title="删除" className="text-destructive" onClick={() => void mutate(item, "delete")}><Trash2 className="h-4 w-4" /></Button></div></div></div>)}</div>
    {audit.length > 0 && <section className="rounded-xl border bg-muted/30 p-4"><h2 className="text-sm font-semibold">审计记录</h2><div className="mt-3 space-y-2">{audit.map((entry) => <div key={entry.id} className="text-xs"><span className="font-medium">{entry.action}</span><span className="ml-2 text-muted-foreground">{date(entry.created_at)} · request_id {entry.request_id}</span></div>)}</div></section>}
  </div></div>
}
