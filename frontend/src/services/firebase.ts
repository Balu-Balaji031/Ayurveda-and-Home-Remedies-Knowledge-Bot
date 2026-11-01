import { getApp, getApps, initializeApp } from 'firebase/app'
import { getAnalytics, isSupported } from 'firebase/analytics'
import { getFirestore } from 'firebase/firestore'
import {
  createUserWithEmailAndPassword,
  getAuth,
  onAuthStateChanged,
  sendPasswordResetEmail,
  signOut,
  signInWithEmailAndPassword,
  updateProfile,
  type User,
} from 'firebase/auth'

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
}

export const firebaseApp = getApps().length ? getApp() : initializeApp(firebaseConfig)
export const auth = getAuth(firebaseApp)
export const db = getFirestore(firebaseApp)

export const analytics = isSupported()
  .then((supported) => supported ? getAnalytics(firebaseApp) : null)
  .catch(() => null)

export async function registerUser(name: string, email: string, password: string) {
  const credential = await createUserWithEmailAndPassword(auth, email, password)
  try {
    await updateProfile(credential.user, { displayName: name })
    const { saveUserProfile } = await import('./userProfiles')
    await saveUserProfile(credential.user, name)
  } finally {
    await signOut(auth)
  }
}

export function loginUser(email: string, password: string) {
  return signInWithEmailAndPassword(auth, email, password)
}

export function resetUserPassword(email: string) {
  return sendPasswordResetEmail(auth, email)
}

export function subscribeToAuthChanges(onChange: (user: User | null) => void) {
  return onAuthStateChanged(auth, onChange)
}

export function getFirebaseErrorMessage(error: unknown) {
  const code = typeof error === 'object' && error !== null && 'code' in error
    ? String(error.code)
    : ''

  switch (code) {
    case 'auth/invalid-email':
      return 'Enter a valid email address.'
    case 'auth/email-already-in-use':
      return 'An account already exists for this email. Try signing in.'
    case 'auth/weak-password':
      return 'Choose a password with at least 6 characters.'
    case 'auth/invalid-credential':
    case 'auth/user-not-found':
    case 'auth/wrong-password':
      return 'The email address or password is incorrect.'
    case 'auth/operation-not-allowed':
      return 'Email and password sign-in is not enabled in Firebase Authentication.'
    case 'auth/network-request-failed':
      return 'Could not reach Firebase. Check your connection and try again.'
    case 'auth/too-many-requests':
      return 'Too many attempts. Wait a moment before trying again.'
    case 'auth/invalid-action-code':
      return 'This password reset link is invalid or has expired.'
    case 'permission-denied':
      return 'Your account could not be saved to Firestore. Check the profile security rules and try again.'
    case 'unavailable':
      return 'Firebase is temporarily unavailable. Check your connection and try again.'
    default:
      return 'Something went wrong. Please try again.'
  }
}