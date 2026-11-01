import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from './useAuth'
import { auth } from '../services/firebase'

export default function ProtectedRoute() {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) {
    return <div className="grid min-h-screen place-items-center bg-[#f5f3eb] text-sm text-[#53665a]">Checking your session...</div>
  }

  return user || auth.currentUser
    ? <Outlet />
    : <Navigate to="/login" replace state={{ from: location }} />
}