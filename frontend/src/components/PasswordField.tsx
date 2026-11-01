import { useState } from 'react'

type PasswordFieldProps = {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  autoComplete: string
  minLength?: number
}

export default function PasswordField({ id, label, value, onChange, autoComplete, minLength }: PasswordFieldProps) {
  const [visible, setVisible] = useState(false)

  return (
    <div>
      <label htmlFor={id} className="mb-2 block text-sm font-medium text-[#344a3a]">{label}</label>
      <div className="relative">
        <input
          id={id}
          name={id}
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          autoComplete={autoComplete}
          minLength={minLength}
          required
          className="auth-input pr-16"
        />
        <button
          type="button"
          onClick={() => setVisible((current) => !current)}
          aria-label={visible ? `Hide ${label.toLowerCase()}` : `Show ${label.toLowerCase()}`}
          aria-pressed={visible}
          className="absolute inset-y-0 right-0 px-3 text-xs font-medium text-[#596d5b] underline decoration-[#aab6a5] underline-offset-4 hover:text-[#244d3a] focus-visible:outline focus-visible:outline-2 focus-visible:outline-inset focus-visible:outline-[#31543c]"
        >
          {visible ? 'Hide' : 'Show'}
        </button>
      </div>
    </div>
  )
}