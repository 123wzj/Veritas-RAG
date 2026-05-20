import * as React from "react"
import { cn } from "@/lib/utils"

interface TabsProps {
  defaultValue: string
  children: React.ReactNode
  className?: string
}

interface TabsListProps {
  children: React.ReactNode
  className?: string
}

interface TabsTriggerProps {
  value: string
  children: React.ReactNode
  className?: string
}

interface TabsContentProps {
  value: string
  children: React.ReactNode
  className?: string
}

const TabsContext = React.createContext<{
  activeTab: string
  setActiveTab: (value: string) => void
}>({ activeTab: "", setActiveTab: () => {} })

const Tabs = ({ defaultValue, children, className }: TabsProps) => {
  const [activeTab, setActiveTab] = React.useState(defaultValue)

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  )
}

const TabsList = ({ children, className }: TabsListProps) => (
  <div className={cn("inline-flex h-10 items-center justify-center rounded-md bg-muted p-1", className)}>
    {children}
  </div>
)

const TabsTrigger = ({ value, children, className }: TabsTriggerProps) => {
  const { activeTab, setActiveTab } = React.useContext(TabsContext)
  const isActive = activeTab === value

  return (
    <button
      onClick={() => setActiveTab(value)}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        isActive && "bg-background text-foreground shadow-sm",
        !isActive && "text-muted-foreground hover:bg-background/50",
        className
      )}
    >
      {children}
    </button>
  )
}

const TabsContent = ({ value, children, className }: TabsContentProps) => {
  const { activeTab } = React.useContext(TabsContext)

  if (activeTab !== value) return null

  return (
    <div className={cn("mt-2 ring-offset-background focus-visible:outline-none", className)}>
      {children}
    </div>
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }
