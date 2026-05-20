import { create } from "zustand"
import { persist } from "zustand/middleware"
import { type User, type UserProfile } from "@/types"

interface UserStore {
  // 用户信息
  user: User | null
  profile: UserProfile | null

  // 认证状态
  isAuthenticated: boolean
  isLoading: boolean

  // Actions
  setUser: (user: User | null) => void
  setProfile: (profile: UserProfile | null) => void
  logout: () => void
}

export const useUserStore = create<UserStore>()(
  persist(
    (set) => ({
      user: null,
      profile: null,
      isAuthenticated: false,
      isLoading: false,

      setUser: (user) =>
        set({
          user,
          isAuthenticated: !!user,
        }),

      setProfile: (profile) => set({ profile }),

      logout: () =>
        set({
          user: null,
          profile: null,
          isAuthenticated: false,
        }),
    }),
    {
      name: "user-storage",
      partialize: (state) => ({
        user: state.user,
        profile: state.profile,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)
