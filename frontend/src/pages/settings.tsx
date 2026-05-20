import { useEffect, useState } from "react"
import { Bot, Heart, User } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { userService } from "@/services/user"
import { useUserStore } from "@/stores/user"

export function SettingsPage() {
  const { user, profile, setUser, setProfile } = useUserStore()
  const [saving, setSaving] = useState(false)
  const [preferredLanguage, setPreferredLanguage] = useState("zh-CN")
  const [interactionStyle, setInteractionStyle] = useState<"concise" | "detailed" | "friendly">("concise")
  const [interests, setInterests] = useState<string[]>([])
  const [interestInput, setInterestInput] = useState("")

  useEffect(() => {
    void loadUserData()
  }, [])

  useEffect(() => {
    if (!profile) return
    setPreferredLanguage(profile.preferred_language || "zh-CN")
    setInteractionStyle(profile.interaction_style || "concise")
    setInterests(profile.interests || [])
  }, [profile])

  const loadUserData = async () => {
    try {
      const [userData, profileData] = await Promise.all([
        userService.getCurrentUser(),
        userService.getProfile(),
      ])
      setUser(userData)
      setProfile(profileData)
    } catch (error) {
      console.error("Failed to load user data:", error)
    }
  }

  const handleSaveProfile = async () => {
    setSaving(true)
    try {
      const updatedProfile = await userService.updateProfile({
        preferred_language: preferredLanguage,
        interaction_style: interactionStyle,
        interests,
        frequently_asked_topics: [],
      })
      setProfile(updatedProfile)
    } catch (error) {
      console.error("Failed to save profile:", error)
      window.alert("保存失败，请稍后重试。")
    } finally {
      setSaving(false)
    }
  }

  const handleAddInterest = () => {
    const next = interestInput.trim()
    if (!next || interests.includes(next)) return
    setInterests([...interests, next])
    setInterestInput("")
  }

  return (
    <div className="min-h-[calc(100vh-57px)] p-4 lg:min-h-screen">
      <div className="quiet-panel mx-auto max-w-5xl rounded-2xl p-5">
        <div className="mb-6">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Settings</p>
          <h1 className="mt-1 text-2xl font-semibold">设置</h1>
          <p className="mt-2 text-sm text-muted-foreground">管理用户信息、交互偏好和记忆策略。</p>
        </div>

        <Tabs defaultValue="profile" className="space-y-5">
          <TabsList>
            <TabsTrigger value="profile">
              <User className="mr-2 h-4 w-4" />
              个人信息
            </TabsTrigger>
            <TabsTrigger value="preferences">
              <Bot className="mr-2 h-4 w-4" />
              AI 偏好
            </TabsTrigger>
            <TabsTrigger value="memory">
              <Heart className="mr-2 h-4 w-4" />
              记忆管理
            </TabsTrigger>
          </TabsList>

          <TabsContent value="profile">
            <Card>
              <CardHeader>
                <CardTitle>个人信息</CardTitle>
                <CardDescription>当前项目使用默认开发用户，后续可以接入正式认证。</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>用户名</Label>
                  <Input value={user?.username || ""} disabled />
                </div>
                <div className="space-y-2">
                  <Label>邮箱</Label>
                  <Input type="email" value={user?.email || ""} placeholder="未设置邮箱" disabled />
                </div>
                <p className="text-sm text-muted-foreground">账户 ID：{user?.id ?? "-"}</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="preferences">
            <Card>
              <CardHeader>
                <CardTitle>AI 交互偏好</CardTitle>
                <CardDescription>这些配置会作为记忆上下文进入 Agentic RAG 流程。</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>偏好语言</Label>
                  <Select value={preferredLanguage} onChange={(event) => setPreferredLanguage(event.target.value)}>
                    <option value="zh-CN">简体中文</option>
                    <option value="en-US">English</option>
                    <option value="ja-JP">日本語</option>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>回答风格</Label>
                  <Select value={interactionStyle} onChange={(event) => setInteractionStyle(event.target.value as any)}>
                    <option value="concise">简洁直接</option>
                    <option value="detailed">详细说明</option>
                    <option value="friendly">友好对话</option>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {interactionStyle === "concise" && "适合快速查询，优先给结论。"}
                    {interactionStyle === "detailed" && "适合学习和复盘，会补充背景与步骤。"}
                    {interactionStyle === "friendly" && "适合连续对话，语气更自然。"}
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>兴趣领域</Label>
                  <div className="flex gap-2">
                    <Input
                      value={interestInput}
                      onChange={(event) => setInterestInput(event.target.value)}
                      onKeyDown={(event) => event.key === "Enter" && handleAddInterest()}
                      placeholder="输入兴趣后按回车"
                    />
                    <Button type="button" onClick={handleAddInterest}>添加</Button>
                  </div>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {interests.map((interest) => (
                      <button
                        key={interest}
                        type="button"
                        onClick={() => setInterests(interests.filter((item) => item !== interest))}
                        className="rounded-full bg-secondary px-3 py-1 text-sm text-secondary-foreground"
                      >
                        {interest} ×
                      </button>
                    ))}
                  </div>
                </div>

                <Button onClick={handleSaveProfile} disabled={saving}>
                  {saving ? "保存中..." : "保存设置"}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="memory">
            <Card>
              <CardHeader>
                <CardTitle>记忆管理</CardTitle>
                <CardDescription>记忆会帮助多轮对话保持连续，但不应该替代知识库证据。</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-2xl border border-border bg-muted/45 p-4">
                  <p className="text-sm font-medium">常问主题</p>
                  <p className="mt-2 text-2xl font-semibold">{profile?.frequently_asked_topics?.length || 0}</p>
                  <p className="mt-1 text-xs text-muted-foreground">来自历史会话的轻量统计。</p>
                </div>
                <div className="rounded-2xl border border-border bg-muted/45 p-4">
                  <p className="text-sm font-medium">长期偏好</p>
                  <p className="mt-2 text-2xl font-semibold">{interests.length}</p>
                  <p className="mt-1 text-xs text-muted-foreground">会参与问题改写和回答风格选择。</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
