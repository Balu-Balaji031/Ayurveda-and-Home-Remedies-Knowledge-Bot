import { useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import AuthLayout from '../components/AuthLayout'
import { getFirebaseErrorMessage, resetUserPassword } from '../services/firebase'

export default function ForgotPassword() {
  const [email, setEmail] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [sent, setSent] = useState(false)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setSent(false)
    setLoading(true)
    try {
      await resetUserPassword(email.trim())
      setSent(true)
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
        {sent && <p role="status" className="rounded-sm border-l-2 border-[#587b5f] bg-[#e8eee4] px-3 py-2 text-sm leading-5 text-[#31543c]">If an account uses this email, a password reset link is on its way.</p>}
        {error && <p role="alert" className="rounded-sm border-l-2 border-[#a75f46] bg-[#f8eae3] px-3 py-2 text-sm leading-5 text-[#793d2e]">{error}</p>}
        <button type="submit" disabled={loading} className="auth-submit">
          {loading ? 'Sending link…' : 'Reset password'}
        </button>
      </form>
      <p className="mt-6 text-center text-sm text-[#697769]">
        <Link to="/login" className="font-semibold text-[#31543c] underline decoration-[#aab6a5] underline-offset-4 hover:text-[#20392e]">Back to Login</Link>
      </p>
    </AuthLayout>
  )
}