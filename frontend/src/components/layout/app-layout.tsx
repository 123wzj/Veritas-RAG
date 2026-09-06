import { useLocation, Link } from "react-router-dom"
import { BookOpen, Brain, Database, MessageSquare, Settings } from "lucide-react"
import type { ReactNode } from "react"

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

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="z-30 shrink-0 border-b border-border bg-white/95 shadow-[0_1px_10px_rgba(15,54,70,0.04)] backdrop-blur">
        <div className="mx-auto flex min-h-14 w-full max-w-[1800px] items-center gap-4 px-4 lg:px-6">
          <Link to="/chat" className="flex min-w-0 shrink-0 items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <BookOpen className="h-4 w-4" />
            </div>
            <div className="hidden min-w-0 sm:block">
              <p className="truncate text-sm font-semibold">Veritas RAG</p>
              <p className="truncate text-[11px] text-muted-foreground">Evidence Agent</p>
            </div>
          </Link>
          <nav className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto" aria-label="主菜单">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  aria-current={isActive ? "page" : undefined}
                  className={cn(
                    "inline-flex h-9 shrink-0 items-center gap-2 rounded-lg px-3 text-sm font-medium transition-colors",
                    isActive ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </Link>
              )
            })}
          </nav>
        </div>
      </header>
      <main className="min-h-0 flex-1">{children}</main>
    </div>
  )
}
