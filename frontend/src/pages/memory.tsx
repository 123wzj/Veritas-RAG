import { useEffect, useState } from "react"
import { Brain, Check, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { apiClient } from "@/services/api"

type Memory = { memory_id: string; memory_type: string; content: string; scope_type: string; confidence: number; status: string }

export function MemoryPage() {
  const [items, setItems] = useState<Memory[]>([])
  const [loading, setLoading] = useState(true)
  const load = async () => { setLoading(true); try { const result = await apiClient.get<{ items: Memory[] }>("/memory/long-term"); setItems(result.items || []) } finally { setLoading(false) } }
  useEffect(() => { void load() }, [])
  const update = async (id: string, status: string) => { await apiClient.patch(`/memory/long-term/${id}`, { status }); await load() }
  return <div className="min-h-screen p-5 lg:p-8"><div className="mx-auto max-w-4xl space-y-6">
    <header><p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Memory</p><h1 className="mt-1 text-3xl font-semibold">更懂你的助手</h1><p className="mt-2 text-sm text-muted-foreground">只保留你确认过的偏好、约束和项目状态。任何记忆都可以撤销。</p></header>
    <Card><CardHeader><CardTitle className="flex items-center gap-2"><Brain className="h-5 w-5 text-primary" />长期记忆</CardTitle></CardHeader><CardContent className="space-y-3">{loading ? <p className="text-sm text-muted-foreground">正在加载…</p> : items.length === 0 ? <p className="text-sm text-muted-foreground">还没有长期记忆。聊天中确认的偏好会出现在这里。</p> : items.map((item) => <div key={item.memory_id} className="flex items-start justify-between gap-4 rounded-xl border p-4"><div className="min-w-0"><div className="flex flex-wrap gap-2"><Badge variant="secondary">{item.memory_type}</Badge><Badge variant="outline">{item.scope_type}</Badge><span className="text-xs text-muted-foreground">置信度 {Math.round(item.confidence * 100)}%</span></div><p className="mt-2 text-sm">{item.content}</p></div><div className="flex shrink-0 gap-1"><Button size="icon" variant="ghost" title="确认" onClick={() => void update(item.memory_id, "active")}><Check className="h-4 w-4" /></Button><Button size="icon" variant="ghost" className="text-destructive" title="删除" onClick={() => void update(item.memory_id, "deleted")}><Trash2 className="h-4 w-4" /></Button></div></div>)}</CardContent></Card>
  </div></div>
}
