import { useEffect, useMemo, useState } from "react"
import { Database, FileText, Loader2, Plus, Trash2, UploadCloud } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { knowledgeService } from "@/services/knowledge"
import { useKnowledgeStore } from "@/stores/knowledge"
import { cn } from "@/lib/utils"

export function KnowledgePage() {
  const {
    knowledgeBases,
    currentKbId,
    documents,
    isLoading,
    setKnowledgeBases,
    addKnowledgeBase,
    deleteKnowledgeBase,
    setCurrentKbId,
    setDocuments,
    setLoading,
  } = useKnowledgeStore()

  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [newKbName, setNewKbName] = useState("")
  const [newKbDesc, setNewKbDesc] = useState("")
  const [uploadingFiles, setUploadingFiles] = useState<Record<string, number>>({})

  const currentKb = useMemo(
    () => knowledgeBases.find((kb) => kb.id === currentKbId),
    [knowledgeBases, currentKbId]
  )
  const completedCount = documents.filter((doc) => doc.status === "completed").length
  const chunkCount = documents.reduce((sum, doc) => sum + (doc.total_chunks || 0), 0)

  useEffect(() => {
    void loadKnowledgeBases()
  }, [])

  const loadKnowledgeBases = async () => {
    setLoading(true)
    try {
      const kbs = await knowledgeService.getKnowledgeBases()
      setKnowledgeBases(kbs)
      if (kbs.length === 0) {
        setCurrentKbId(null)
        setDocuments([])
        return
      }
      const selectedKbId = currentKbId && kbs.some((kb) => kb.id === currentKbId) ? currentKbId : kbs[0].id
      setCurrentKbId(selectedKbId)
      await loadDocuments(selectedKbId)
    } catch (error) {
      console.error("Failed to load knowledge bases:", error)
    } finally {
      setLoading(false)
    }
  }

  const loadDocuments = async (kbId: number) => {
    setLoading(true)
    try {
      setDocuments(await knowledgeService.getDocuments(kbId))
    } catch (error) {
      console.error("Failed to load documents:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreateKb = async () => {
    if (!newKbName.trim()) return
    try {
      const kb = await knowledgeService.createKnowledgeBase({
        name: newKbName.trim(),
        description: newKbDesc.trim() || undefined,
      })
      addKnowledgeBase(kb)
      setCurrentKbId(kb.id)
      setDocuments([])
      setShowCreateDialog(false)
      setNewKbName("")
      setNewKbDesc("")
    } catch (error) {
      console.error("Failed to create knowledge base:", error)
      window.alert("创建知识库失败，请稍后重试。")
    }
  }

  const handleDeleteKb = async (kbId: number) => {
    if (!window.confirm("确定删除这个知识库吗？其中的文档和索引也会被删除。")) return
    try {
      await knowledgeService.deleteKnowledgeBase(kbId)
      deleteKnowledgeBase(kbId)
      const remaining = useKnowledgeStore.getState().knowledgeBases.filter((kb) => kb.id !== kbId)
      if (remaining.length > 0) {
        setCurrentKbId(remaining[0].id)
        await loadDocuments(remaining[0].id)
      } else {
        setCurrentKbId(null)
        setDocuments([])
      }
    } catch (error) {
      console.error("Failed to delete knowledge base:", error)
      window.alert("删除知识库失败，请稍后重试。")
    }
  }

  const handleSelectKb = async (kbId: number) => {
    setCurrentKbId(kbId)
    await loadDocuments(kbId)
  }

  const handleFileUpload = async (kbId: number, files: FileList) => {
    const fileArray = Array.from(files)
    for (const file of fileArray) {
      setUploadingFiles((prev) => ({ ...prev, [file.name]: 0 }))
    }

    try {
      for (const file of fileArray) {
        const upload = (confirmSameName = false) =>
          knowledgeService.uploadDocument(
            kbId,
            file,
            (progress) => setUploadingFiles((prev) => ({ ...prev, [file.name]: progress })),
            { confirmSameName }
          )

        try {
          await upload()
        } catch (error) {
          const uploadError = error as Error & { status?: number; detail?: any }
          const detail = uploadError.detail
          if (uploadError.status === 409 && detail?.code === "same_filename") {
            if (!window.confirm(detail.message || `知识库中已存在同名文件「${file.name}」，是否继续上传？`)) continue
            await upload(true)
            continue
          }
          if (uploadError.status === 409 && detail?.code === "duplicate_file") {
            window.alert(detail.message || `文件「${file.name}」已上传过，请勿重复上传。`)
            continue
          }
          throw error
        }
      }
      await loadDocuments(kbId)
      setShowUploadDialog(false)
    } catch (error) {
      console.error("Failed to upload documents:", error)
      window.alert("上传文件失败，请检查文件格式或稍后重试。")
    } finally {
      setUploadingFiles({})
    }
  }

  const handleDeleteDocument = async (kbId: number, docId: string, filename: string) => {
    if (!window.confirm(`确定删除文档「${filename}」吗？`)) return
    try {
      await knowledgeService.deleteDocument(kbId, docId)
      await loadDocuments(kbId)
    } catch (error) {
      console.error("Failed to delete document:", error)
      window.alert("删除文档失败，请稍后重试。")
    }
  }

  return (
    <div className="flex h-[calc(100vh-57px)] min-h-0 lg:h-screen">
      <aside className="hidden w-80 shrink-0 border-r border-border bg-white/86 p-4 backdrop-blur lg:block">
        <Button className="mb-4 w-full justify-start rounded-xl" onClick={() => setShowCreateDialog(true)}>
          <Plus className="mr-2 h-4 w-4" />
          新建知识库
        </Button>
        <div className="space-y-2">
          {knowledgeBases.map((kb) => (
            <button
              key={kb.id}
              type="button"
              onClick={() => void handleSelectKb(kb.id)}
              className={cn(
                "w-full rounded-2xl border p-4 text-left transition-colors",
                currentKbId === kb.id ? "border-primary bg-primary/8" : "border-border bg-white hover:bg-muted/60"
              )}
            >
              <div className="flex items-center gap-3">
                <Database className="h-4 w-4 text-primary" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold">{kb.name}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{kb.document_count || 0} 个文档</p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </aside>

      <main className="min-w-0 flex-1 p-4">
        <div className="quiet-panel flex h-full flex-col rounded-2xl">
          <header className="border-b border-border p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Knowledge base</p>
                <h1 className="mt-1 text-2xl font-semibold">{currentKb?.name || "选择或创建知识库"}</h1>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
                  文档会被解析为父子块，子块写入 Dense 向量和 BM25 sparse，回答时再回补父块上下文。
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" className="rounded-xl lg:hidden" onClick={() => setShowCreateDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  新建
                </Button>
                {currentKb && (
                  <>
                    <Button className="rounded-xl" onClick={() => setShowUploadDialog(true)}>
                      <UploadCloud className="mr-2 h-4 w-4" />
                      上传文件
                    </Button>
                    <Button variant="outline" className="rounded-xl text-destructive" onClick={() => void handleDeleteKb(currentKb.id)}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </>
                )}
              </div>
            </div>

            {currentKb && (
              <div className="mt-5 grid gap-3 sm:grid-cols-3">
                <Stat label="文档" value={documents.length} />
                <Stat label="已完成" value={completedCount} />
                <Stat label="子块" value={chunkCount} />
              </div>
            )}
          </header>

          <section className="min-h-0 flex-1 overflow-auto p-5">
            {isLoading ? (
              <div className="flex h-full items-center justify-center text-muted-foreground">
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                正在加载
              </div>
            ) : currentKb ? (
              documents.length > 0 ? (
                <div className="overflow-hidden rounded-2xl border border-border bg-white">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>文件名</TableHead>
                        <TableHead>类型</TableHead>
                        <TableHead>状态</TableHead>
                        <TableHead>Chunks</TableHead>
                        <TableHead>上传时间</TableHead>
                        <TableHead className="text-right">操作</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {documents.map((doc) => (
                        <TableRow key={doc.id}>
                          <TableCell className="max-w-[360px] truncate font-medium">{doc.filename}</TableCell>
                          <TableCell>{getModalityBadge(doc.modality)}</TableCell>
                          <TableCell>{getStatusBadge(doc.status)}</TableCell>
                          <TableCell>{doc.total_chunks}</TableCell>
                          <TableCell>{new Date(doc.created_at).toLocaleDateString()}</TableCell>
                          <TableCell className="text-right">
                            <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={() => void handleDeleteDocument(currentKb.id, doc.doc_id, doc.filename)}>
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <EmptyState onUpload={() => setShowUploadDialog(true)} />
              )
            ) : (
              <div className="flex h-full items-center justify-center">
                <div className="max-w-md text-center">
                  <Database className="mx-auto h-12 w-12 text-primary" />
                  <h3 className="mt-4 text-lg font-semibold">还没有知识库</h3>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">创建一个知识库后，就可以上传文档并测试 RAG 检索链路。</p>
                  <Button className="mt-5 rounded-xl" onClick={() => setShowCreateDialog(true)}>
                    <Plus className="mr-2 h-4 w-4" />
                    创建知识库
                  </Button>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>

      {Object.keys(uploadingFiles).length > 0 && (
        <div className="fixed bottom-4 right-4 z-50 w-80 rounded-2xl border border-border bg-white p-4 shadow-xl">
          <h4 className="mb-3 text-sm font-semibold">上传中</h4>
          <div className="space-y-3">
            {Object.entries(uploadingFiles).map(([name, progress]) => (
              <div key={name}>
                <div className="mb-1 flex items-center justify-between gap-3 text-xs">
                  <span className="truncate">{name}</span>
                  <span className="text-muted-foreground">{progress}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${progress}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>创建知识库</DialogTitle>
            <DialogDescription>为一类资料创建独立索引，便于后续检索和权限隔离。</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <label className="text-sm font-medium">名称</label>
              <Input value={newKbName} onChange={(event) => setNewKbName(event.target.value)} placeholder="例如：Agentic RAG 学习资料" />
            </div>
            <div>
              <label className="text-sm font-medium">描述</label>
              <Textarea value={newKbDesc} onChange={(event) => setNewKbDesc(event.target.value)} placeholder="可选，说明这个知识库的用途" rows={3} />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>取消</Button>
            <Button onClick={handleCreateKb}>创建</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showUploadDialog} onOpenChange={setShowUploadDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>上传文件</DialogTitle>
            <DialogDescription>支持 PDF、DOCX、PPTX、Markdown、TXT、HTML 与图片等格式。</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <label className="flex h-36 w-full cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border bg-muted/45 transition-colors hover:bg-muted">
              <UploadCloud className="mb-3 h-8 w-8 text-primary" />
              <p className="text-sm font-medium">点击选择文件</p>
              <p className="mt-1 text-xs text-muted-foreground">可多选，重复文件会自动提示</p>
              <input
                type="file"
                className="hidden"
                multiple
                accept=".pdf,.docx,.pptx,.md,.html,.txt,.png,.jpg,.jpeg"
                onChange={(event) => {
                  if (event.target.files && event.target.files.length > 0 && currentKbId) {
                    void handleFileUpload(currentKbId, event.target.files)
                  }
                }}
              />
            </label>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-2xl border border-border bg-muted/45 px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
    </div>
  )
}

function EmptyState({ onUpload }: { onUpload: () => void }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="max-w-md text-center">
        <FileText className="mx-auto h-12 w-12 text-primary" />
        <h3 className="mt-4 text-lg font-semibold">暂无文档</h3>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">上传第一批文档后，系统会自动解析、分块、向量化并建立检索索引。</p>
        <Button className="mt-5 rounded-xl" onClick={onUpload}>
          <UploadCloud className="mr-2 h-4 w-4" />
          上传文件
        </Button>
      </div>
    </div>
  )
}

function getStatusBadge(status: string) {
  const variants: Record<string, "default" | "secondary" | "destructive" | "success" | "warning"> = {
    pending: "warning",
    processing: "default",
    completed: "success",
    failed: "destructive",
  }
  const labels: Record<string, string> = {
    pending: "等待中",
    processing: "处理中",
    completed: "已完成",
    failed: "失败",
  }
  return <Badge variant={variants[status] ?? "secondary"}>{labels[status] ?? status}</Badge>
}

function getModalityBadge(modality: string) {
  const labels: Record<string, string> = {
    text: "文本",
    image: "图片",
    table: "表格",
    mixed: "混合",
  }
  return <Badge variant="secondary">{labels[modality] ?? modality}</Badge>
}
