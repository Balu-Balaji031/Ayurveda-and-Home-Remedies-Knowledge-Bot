import type { User } from 'firebase/auth'
import { doc, runTransaction, serverTimestamp } from 'firebase/firestore'
import { db } from './firebase'

export async function saveUserProfile(user: User, name: string) {
  const profileRef = doc(db, 'users', user.uid)

  await runTransaction(db, async (transaction) => {
    const existingProfile = await transaction.get(profileRef)
    if (existingProfile.exists()) return

    transaction.set(profileRef, {
      uid: user.uid,
      name,
      email: user.email ?? '',
      createdAt: serverTimestamp(),
    })
  })
}

export function ensureUserProfile(user: User) {
  const name = user.displayName?.trim() || user.email?.split('@')[0] || 'AyurBot user'
  return saveUserProfile(user, name)
}