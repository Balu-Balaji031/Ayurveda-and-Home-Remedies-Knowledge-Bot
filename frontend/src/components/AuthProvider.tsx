import { useEffect, useState, type ReactNode } from 'react'
import type { User } from 'firebase/auth'
import { subscribeToAuthChanges } from '../services/firebase'
import { AuthContext } from './AuthContext'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => subscribeToAuthChanges((nextUser) => {
    setUser(nextUser)
    setLoading(false)
  }), [])

  return (
    <AuthContext.Provider value={{ user, loading }}>
      {children}
    </AuthContext.Provider>
  )
}