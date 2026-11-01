import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthLayout from '../components/AuthLayout'
import PasswordField from '../components/PasswordField'
import { getFirebaseErrorMessage, registerUser } from '../services/firebase'

export default function Register() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    if (!name.trim() || !email.trim() || !password || !confirmPassword) {
      setError('Complete all fields before creating your account.')
      return
    }
    if (password !== confirmPassword) {
      setError('Your passwords do not match. Please check them and try again.')
      return
    }
    setLoading(true)
    try {
      await registerUser(name.trim(), email.trim(), password)
      navigate('/login', {
        replace: true,
        state: { message: 'Your account has been created. Log in to continue to AyurBot.' },
      })
    } catch (authError) {
      setError(getFirebaseErrorMessage(authError))
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="full-name" className="mb-2 block text-sm font-medium text-[#344a3a]">Full name</label>
          <input id="full-name" name="name" type="text" autoComplete="name" required value={name} onChange={(event) => setName(event.target.value)} className="auth-input" />
        </div>
        <div>
          <label htmlFor="email" className="mb-2 block text-sm font-medium text-[#344a3a]">Email address</label>
          <input id="email" name="email" type="email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} className="auth-input" />
        </div>
        <PasswordField id="password" label="Set password" value={password} onChange={setPassword} autoComplete="new-password" minLength={6} />
        <PasswordField id="confirm-password" label="Confirm password" value={confirmPassword} onChange={setConfirmPassword} autoComplete="new-password" minLength={6} />
        {error && <p role="alert" className="rounded-sm border-l-2 border-[#a75f46] bg-[#f8eae3] px-3 py-2 text-sm leading-5 text-[#793d2e]">{error}</p>}
        <button type="submit" disabled={loading} className="auth-submit !mt-5">
          {loading ? 'Creating account…' : 'Create account'}
        </button>
      </form>
      <p className="mt-6 text-center text-sm text-[#697769]">
        Already have an account? <Link to="/login" className="font-semibold text-[#31543c] underline decoration-[#aab6a5] underline-offset-4 hover:text-[#20392e]">Log in</Link>
      </p>
    </AuthLayout>
  )
}