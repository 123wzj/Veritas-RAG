import { cn } from "@/lib/utils"

interface ToastProps {
  title?: string
  description?: string
  variant?: "default" | "destructive"
  duration?: number
}

const Toast = ({ title, description, variant = "default" }: ToastProps) => {
  return (
    <div
      className={cn(
        "fixed bottom-4 right-4 z-50 rounded-lg border p-4 shadow-lg",
        "bg-background text-foreground",
        variant === "destructive" && "bg-destructive text-destructive-foreground"
      )}
    >
      {title && <div className="font-semibold">{title}</div>}
      {description && <div className="text-sm opacity-90">{description}</div>}
    </div>
  )
}

export { Toast }
