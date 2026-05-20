import { create } from "zustand"

import { type ChatMessage, type ChatSession, type StreamEvent } from "@/types"

interface ChatStore {
  currentSession: ChatSession | null
  sessions: ChatSession[]
  isStreaming: boolean
  currentRequestId: string | null
  setCurrentSession: (session: ChatSession | null) => void
  createSession: (session: ChatSession) => void
  upsertSession: (session: ChatSession) => void
  addMessage: (message: ChatMessage) => void
  setMessages: (messages: ChatMessage[]) => void
  updateLastMessage: (content: string, citations?: any[]) => void
  appendThinkingEventToLastMessage: (event: StreamEvent) => void
  setLastMessageThinkingCollapsed: (collapsed: boolean) => void
  setStreaming: (isStreaming: boolean) => void
  setCurrentRequestId: (requestId: string | null) => void
  deleteSession: (sessionId: string) => void
  setSessions: (sessions: ChatSession[]) => void
  clearCurrentSession: () => void
}

export const useChatStore = create<ChatStore>((set) => ({
  currentSession: null,
  sessions: [],
  isStreaming: false,
  currentRequestId: null,

  setCurrentSession: (session) =>
    set((state) => {
      if (!session) {
        return { currentSession: null }
      }

      const exists = state.sessions.some((item) => item.session_id === session.session_id)
      const sessions = exists
        ? state.sessions.map((item) => (item.session_id === session.session_id ? session : item))
        : [session, ...state.sessions]

      return {
        currentSession: session,
        sessions,
      }
    }),

  createSession: (session) =>
    set((state) => ({
      currentSession: session,
      sessions: [session, ...state.sessions.filter((item) => item.session_id !== session.session_id)],
    })),

  upsertSession: (session) =>
    set((state) => {
      const exists = state.sessions.some((item) => item.session_id === session.session_id)
      const sessions = exists
        ? state.sessions.map((item) => (item.session_id === session.session_id ? session : item))
        : [session, ...state.sessions]

      const currentSession =
        state.currentSession?.session_id === session.session_id
          ? { ...session, messages: state.currentSession.messages }
          : state.currentSession

      return { sessions, currentSession }
    }),

  addMessage: (message) =>
    set((state) => {
      if (!state.currentSession) return state

      const updatedSession = {
        ...state.currentSession,
        messages: [...state.currentSession.messages, message],
        message_count: state.currentSession.message_count + 1,
        last_active: new Date().toISOString(),
      }

      return {
        currentSession: updatedSession,
        sessions: state.sessions.map((item) =>
          item.session_id === updatedSession.session_id ? updatedSession : item
        ),
      }
    }),

  setMessages: (messages) =>
    set((state) => {
      if (!state.currentSession) return state

      const updatedSession = {
        ...state.currentSession,
        messages,
        message_count: messages.length,
      }

      return {
        currentSession: updatedSession,
        sessions: state.sessions.map((item) =>
          item.session_id === updatedSession.session_id ? updatedSession : item
        ),
      }
    }),

  updateLastMessage: (content, citations) =>
    set((state) => {
      if (!state.currentSession) return state

      const messages = [...state.currentSession.messages]
      const lastMessage = messages[messages.length - 1]

      if (lastMessage && lastMessage.role === "assistant") {
        messages[messages.length - 1] = {
          ...lastMessage,
          content,
          citations: citations ?? lastMessage.citations,
        }
      }

      const updatedSession = {
        ...state.currentSession,
        messages,
        last_active: new Date().toISOString(),
      }

      return {
        currentSession: updatedSession,
        sessions: state.sessions.map((item) =>
          item.session_id === updatedSession.session_id ? updatedSession : item
        ),
      }
    }),

  appendThinkingEventToLastMessage: (event) =>
    set((state) => {
      if (!state.currentSession) return state

      const messages = [...state.currentSession.messages]
      const lastMessage = messages[messages.length - 1]
      if (lastMessage && lastMessage.role === "assistant") {
        messages[messages.length - 1] = {
          ...lastMessage,
          thinkingEvents: [...(lastMessage.thinkingEvents || []), event],
          thinkingCollapsed: false,
        }
      }

      const updatedSession = {
        ...state.currentSession,
        messages,
        last_active: new Date().toISOString(),
      }

      return {
        currentSession: updatedSession,
        sessions: state.sessions.map((item) =>
          item.session_id === updatedSession.session_id ? updatedSession : item
        ),
      }
    }),

  setLastMessageThinkingCollapsed: (collapsed) =>
    set((state) => {
      if (!state.currentSession) return state

      const messages = [...state.currentSession.messages]
      const lastMessage = messages[messages.length - 1]
      if (lastMessage && lastMessage.role === "assistant") {
        messages[messages.length - 1] = {
          ...lastMessage,
          thinkingCollapsed: collapsed,
        }
      }

      const updatedSession = { ...state.currentSession, messages }
      return {
        currentSession: updatedSession,
        sessions: state.sessions.map((item) =>
          item.session_id === updatedSession.session_id ? updatedSession : item
        ),
      }
    }),

  setStreaming: (isStreaming) => set({ isStreaming }),
  setCurrentRequestId: (requestId) => set({ currentRequestId: requestId }),

  deleteSession: (sessionId) =>
    set((state) => ({
      sessions: state.sessions.filter((item) => item.session_id !== sessionId),
      currentSession:
        state.currentSession?.session_id === sessionId ? null : state.currentSession,
    })),

  setSessions: (sessions) =>
    set((state) => {
      const matchedCurrent = state.currentSession
        ? sessions.find((item) => item.session_id === state.currentSession?.session_id)
        : null
      const currentSession =
        matchedCurrent && state.currentSession
          ? { ...matchedCurrent, messages: state.currentSession.messages }
          : state.currentSession
      return { sessions, currentSession }
    }),

  clearCurrentSession: () =>
    set((state) => ({
      currentSession: state.currentSession
        ? { ...state.currentSession, messages: [], message_count: 0 }
        : null,
    })),
}))
