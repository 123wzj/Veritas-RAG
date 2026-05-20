# Veritas RAG - Frontend

基于 React + TypeScript + Vite + shadcn/ui 的 Agentic RAG 问答系统前端应用。

## 技术栈

- **框架**: React 18 + TypeScript
- **构建**: Vite
- **UI 组件**: shadcn/ui (Radix UI + Tailwind CSS)
- **状态管理**: Zustand
- **路由**: React Router (待集成)
- **HTTP**: axios
- **Markdown**: react-markdown
- **图标**: lucide-react

## 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env .env
```

### 3. 启动开发服务器

```bash
npm run dev
```

### 4. 构建生产版本

```bash
npm run build
```

## 功能模块

### 聊天界面
- SSE 流式输出
- Markdown 渲染
- 引用展示
- 多轮对话
- 会话历史

### 知识库管理
- 知识库列表
- 创建/编辑/删除知识库
- 文档上传
- 文档列表
- 索引状态

### 用户设置
- 用户画像
- 偏好设置
- 记忆管理

## 开发状态

### ✅ 已完成
- 项目骨架
- UI 基础组件 (Button, Input, Card, etc.)
- 聊天界面
- SSE 流式输出集成
- Zustand 状态管理
- API 服务层
- TypeScript 类型定义

### 🚧 待开发
- 知识库管理页面
- 用户设置页面
- 路由集成 (React Router)
- 主题切换 (暗色模式)
- 更多 UI 组件
- 错误处理与提示
- 响应式布局优化

## 与后端对接

后端 API 地址通过环境变量 `VITE_API_BASE_URL` 配置。
