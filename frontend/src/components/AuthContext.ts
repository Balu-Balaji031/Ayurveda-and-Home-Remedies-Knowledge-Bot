import { createContext } from 'react'
import type { User } from 'firebase/auth'

export type AuthContextValue = {
  user: User | null
  loading: boolean
}

export const AuthContext = createContext<AuthContextValue | null>(null)