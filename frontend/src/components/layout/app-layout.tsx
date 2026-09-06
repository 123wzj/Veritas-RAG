import { useEffect, useState, type ReactNode } from "react"
import { Link, useLocation } from "react-router-dom"
import { BookOpen, Brain, Database, MessageSquare, PanelLeftClose, PanelLeftOpen, Settings } from "lucide-react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface LayoutProps {
  children: ReactNode
}

const navItems = [
  { path: "/chat", icon: MessageSquare, label: "聊天" },
  { path: "/knowledge", icon: Database, label: "知识库" },
  { path: "/memory", icon: Brain, label: "记忆" },
  { path: "/settings", icon: Settings, label: "设置" },
]

export function AppLayout({ children }: LayoutProps) {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem("app-sidebar-collapsed") === "true")

  useEffect(() => {
    localStorage.setItem("app-sidebar-collapsed", String(collapsed))
  }, [collapsed])

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <aside
        className={cn(
          "hidden shrink-0 flex-col border-r border-border bg-white transition-[width] duration-200 lg:flex",
          collapsed ? "w-[68px]" : "w-[260px]"
        )}
      >
        <div className="flex h-14 items-center gap-2 border-b border-border px-3">
          <Link to="/chat" className="flex min-w-0 flex-1 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <BookOpen className="h-4 w-4" />
            </div>
            {!collapsed && (
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold">Veritas RAG</p>
                <p className="truncate text-xs text-muted-foreground">Evidence Agent</p>
              </div>
            )}
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9 rounded-xl"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "展开侧边栏" : "收起侧边栏"}
          >
            {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          </Button>
        </div>

        <nav className="flex-1 space-y-1 p-2">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                title={collapsed ? item.label : undefined}
                className={cn(
                  "flex h-11 items-center gap-3 rounded-xl px-3 text-sm transition-colors",
                  isActive ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/70 hover:text-foreground",
                  collapsed && "justify-center px-0"
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed && <span className="font-medium">{item.label}</span>}
              </Link>
            )
          })}
        </nav>
      </aside>

      <main className="min-w-0 flex-1">
        <div className="border-b border-border bg-white px-3 py-2 lg:hidden">
          <nav className="grid grid-cols-4 gap-2">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    "flex items-center justify-center gap-2 rounded-xl px-3 py-2 text-sm",
                    isActive ? "bg-muted text-foreground" : "text-muted-foreground"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              )
            })}
          </nav>
        </div>
        {children}
      </main>
    </div>
  )
}
