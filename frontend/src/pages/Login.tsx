import { useState, type FormEvent } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import AuthLayout from '../components/AuthLayout'
import PasswordField from '../components/PasswordField'
import { getFirebaseErrorMessage, loginUser } from '../services/firebase'

export default function Login() {
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [message] = useState<string | null>(location.state?.message ?? null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await loginUser(email.trim(), password)
      navigate('/chat', { replace: true })
    } catch (authError) {
      setError(getFirebaseErrorMessage(authError))
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout>
      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label htmlFor="email" className="mb-2 block text-sm font-medium text-[#344a3a]">Email address</label>
          <input id="email" name="email" type="email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} className="auth-input" />
        </div>
        <PasswordField id="password" label="Password" value={password} onChange={setPassword} autoComplete="current-password" />
        <div className="-mt-1 flex justify-end">
          <Link to="/forgot-password" className="text-sm text-[#426348] underline decoration-[#aab6a5] underline-offset-4 hover:text-[#20392e] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#31543c]">Forgot password?</Link>
        </div>
        {message && <p role="status" className="rounded-sm border-l-2 border-[#587b5f] bg-[#e8eee4] px-3 py-2 text-sm leading-5 text-[#31543c]">{message}</p>}
        {error && <p role="alert" className="rounded-sm border-l-2 border-[#a75f46] bg-[#f8eae3] px-3 py-2 text-sm leading-5 text-[#793d2e]">{error}</p>}
        <button type="submit" disabled={loading} className="auth-submit">
          {loading ? 'Signing in…' : 'Log in'}
        </button>
      </form>
      <p className="mt-6 text-center text-sm text-[#697769]">
        New to AyurBot? <Link to="/register" className="font-semibold text-[#31543c] underline decoration-[#aab6a5] underline-offset-4 hover:text-[#20392e]">Create an account</Link>
      </p>
    </AuthLayout>
  )
}