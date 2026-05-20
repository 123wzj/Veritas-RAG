import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"

import { AppLayout } from "./components/layout/app-layout"
import { ChatPage } from "./pages/chat"
import { KnowledgePage } from "./pages/knowledge"
import { SettingsPage } from "./pages/settings"

function App() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppLayout><Navigate to="/chat" replace /></AppLayout>} />
          <Route path="/chat" element={<AppLayout><ChatPage /></AppLayout>} />
          <Route path="/knowledge" element={<AppLayout><KnowledgePage /></AppLayout>} />
          <Route path="/settings" element={<AppLayout><SettingsPage /></AppLayout>} />
        </Routes>
      </BrowserRouter>
    </div>
  )
}

export default App
