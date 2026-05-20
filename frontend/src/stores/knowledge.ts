import { create } from "zustand"
import { persist } from "zustand/middleware"

import { type Document, type KnowledgeBase } from "@/types"

interface KnowledgeStore {
  knowledgeBases: KnowledgeBase[]
  currentKbId: number | null
  documents: Document[]
  isLoading: boolean
  error: string | null
  setKnowledgeBases: (kbs: KnowledgeBase[]) => void
  addKnowledgeBase: (kb: KnowledgeBase) => void
  updateKnowledgeBase: (kbId: number, updates: Partial<KnowledgeBase>) => void
  deleteKnowledgeBase: (kbId: number) => void
  setCurrentKbId: (kbId: number | null) => void
  setDocuments: (docs: Document[]) => void
  addDocument: (doc: Document) => void
  updateDocument: (docId: string, updates: Partial<Document>) => void
  deleteDocument: (docId: string) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useKnowledgeStore = create<KnowledgeStore>()(
  persist(
    (set) => ({
      knowledgeBases: [],
      currentKbId: null,
      documents: [],
      isLoading: false,
      error: null,

      setKnowledgeBases: (kbs) => set({ knowledgeBases: kbs }),

      addKnowledgeBase: (kb) =>
        set((state) => ({
          knowledgeBases: [...state.knowledgeBases, kb],
        })),

      updateKnowledgeBase: (kbId, updates) =>
        set((state) => ({
          knowledgeBases: state.knowledgeBases.map((kb) =>
            kb.id === kbId ? { ...kb, ...updates } : kb
          ),
        })),

      deleteKnowledgeBase: (kbId) =>
        set((state) => ({
          knowledgeBases: state.knowledgeBases.filter((kb) => kb.id !== kbId),
          currentKbId: state.currentKbId === kbId ? null : state.currentKbId,
        })),

      setCurrentKbId: (kbId) => set({ currentKbId: kbId }),

      setDocuments: (docs) => set({ documents: docs }),

      addDocument: (doc) =>
        set((state) => ({
          documents: [...state.documents, doc],
        })),

      updateDocument: (docId, updates) =>
        set((state) => ({
          documents: state.documents.map((doc) =>
            doc.doc_id === docId ? { ...doc, ...updates } : doc
          ),
        })),

      deleteDocument: (docId) =>
        set((state) => ({
          documents: state.documents.filter((doc) => doc.doc_id !== docId),
        })),

      setLoading: (loading) => set({ isLoading: loading }),
      setError: (error) => set({ error }),
    }),
    {
      name: "knowledge-storage",
      partialize: (state) => ({
        currentKbId: state.currentKbId,
      }),
    }
  )
)
