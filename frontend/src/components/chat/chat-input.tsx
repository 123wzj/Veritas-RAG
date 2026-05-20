import { useState, type ChangeEvent, type KeyboardEvent } from "react"
import { SendHorizonal } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
  placeholder?: string
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = "输入问题，Enter 发送，Shift + Enter 换行",
}: ChatInputProps) {
  const [value, setValue] = useState("")

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue("")
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="rounded-3xl border border-border bg-white p-3 shadow-[0_12px_40px_rgba(15,54,70,0.10)]">
      <div className="flex items-end gap-3">
        <Textarea
          value={value}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="min-h-[54px] max-h-40 resize-none rounded-xl border-0 bg-transparent px-2 py-3 text-sm leading-6 shadow-none placeholder:text-muted-foreground/80 focus-visible:ring-0"
        />
        <Button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          size="icon"
          className="h-11 w-11 shrink-0 rounded-xl"
          aria-label="发送"
        >
          <SendHorizonal className="h-4 w-4" />
        </Button>
      </div>
      <div className="mt-2 flex items-center justify-between px-1 text-xs text-muted-foreground">
        <span>{disabled ? "正在生成回答" : "支持知识库、联网或混合增强"}</span>
        <span>{value.trim().length} 字</span>
      </div>
    </div>
  )
}
