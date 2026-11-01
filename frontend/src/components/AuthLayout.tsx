import type { ReactNode } from 'react'

export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
      <main className="flex min-h-screen items-center justify-center bg-[#f5f3eb] px-5 py-8 text-[#20392e]">
        <section className="w-full max-w-[420px]">
          <header className="mb-8 text-center">
            <h1 className="font-serif text-4xl text-[#244d3a]">AyurBot</h1>
            <p className="mt-3 font-serif text-base leading-6 text-[#667565]">
              "Ancient wisdom, thoughtful guidance for everyday well-being."
            </p>
          </header>
          {children}
        </section>
      </main>
  )
}