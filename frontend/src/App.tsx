import './App.css'
import { Suspense, lazy } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { useAuth } from './components/useAuth'
import ProtectedRoute from './components/ProtectedRoute'
import Login from './pages/Login'
import Register from './pages/Register'
import ForgotPassword from './pages/ForgotPassword'

const Dashboard = lazy(() => import('./pages/Dashboard'))

function App() {
  const { user, loading } = useAuth()

  if (loading) {
    return <div className="grid min-h-screen place-items-center bg-[#f5f3eb] text-sm text-[#53665a]">Preparing your AyurBot experience...</div>
  }

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/chat" replace /> : <Login />} />
      <Route path="/register" element={user ? <Navigate to="/chat" replace /> : <Register />} />
      <Route path="/forgot-password" element={user ? <Navigate to="/chat" replace /> : <ForgotPassword />} />
      <Route element={<ProtectedRoute />}>
        <Route path="/chat" element={<Suspense fallback={<div className="grid min-h-screen place-items-center bg-[#f5f3eb] text-sm text-[#53665a]">Loading chat...</div>}><Dashboard /></Suspense>} />
        <Route path="/dashboard" element={<Suspense fallback={<div className="grid min-h-screen place-items-center bg-[#f5f3eb] text-sm text-[#53665a]">Loading chat...</div>}><Dashboard /></Suspense>} />
      </Route>
      <Route path="*" element={<Navigate to={user ? '/chat' : '/login'} replace />} />
    </Routes>
  )
}

export default App
