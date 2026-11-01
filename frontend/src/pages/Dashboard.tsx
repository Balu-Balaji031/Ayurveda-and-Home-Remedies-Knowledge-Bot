import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { signOut } from 'firebase/auth'
import { Timestamp } from 'firebase/firestore'
import { useNavigate } from 'react-router-dom'
import type { User } from 'firebase/auth'
import { auth } from '../services/firebase'
import {
  observeConversations,
  observeMessages,
  saveUserTurn,
  type ChatMessage,
  type ChatMessageMetadata,
  type Conversation,
} from '../services/chatHistory'
import { useAuth } from '../components/useAuth'
import {
  askRemedyFollowUp,
  ChatApiError,
  explainRemedy,
  recommendRemedies,
  type ChatSource,
  type FollowUpTurn,
  type RemedyCard as RemedyCardData,
} from '../services/chatApi'
import RemedyCard from '../components/RemedyCard'

type VisibleChatMessage = ChatMessage & { sources?: ChatSource[] }
type PendingTurn = {
  content: string
  baseConversationId: string | null
  conversationId: string
  isNewConversation: boolean
  userMessageId: string
  assistantMessageId: string
}

function HumanIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="size-5 fill-none stroke-current" strokeWidth="1.7">
      <circle cx="12" cy="8" r="3.2" />
      <path d="M5.8 20c.5-3.5 2.8-5.4 6.2-5.4s5.7 1.9 6.2 5.4" strokeLinecap="round" />
    </svg>
  )
}

function RobotIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="size-5 fill-none stroke-current" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 4V2.5M9 2.5h6M5 8.5h14v10H5zM8.5 12h.1M15.5 12h.1M9 15.5h6M3 11v5M21 11v5" />
    </svg>
  )
}

function ProfileIcon({ user }: { user: User }) {
  const initials = user.displayName?.trim().split(/\s+/).slice(0, 2).map((part) => part[0]).join('').toUpperCase()
  return (
    <span className="grid size-9 shrink-0 place-items-center rounded-full bg-[#e1e8dc] text-xs font-semibold text-[#31543c]" aria-hidden="true">
      {initials || <HumanIcon />}
    </span>
  )
}

function selectedContext(messages: VisibleChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const metadata = messages[index].metadata
    if (metadata?.selectedRemedy) {
      return { problem: metadata.problem, remedy: metadata.selectedRemedy }
    }
  }
  return null
}

function priorTurns(messages: VisibleChatMessage[]): FollowUpTurn[] {
  const turns: FollowUpTurn[] = []
  for (let index = 0; index < messages.length - 1; index += 1) {
    const userMessage = messages[index]
    const assistantMessage = messages[index + 1]
    if (userMessage.role === 'user' && assistantMessage.role === 'assistant') {
      turns.push({ question: userMessage.content, answer: assistantMessage.content })
      index += 1
    }
  }
  return turns.slice(-8)
}

function recommendationSummary(mode: 'pdf' | 'ai_generated' | 'abstain', label: string | null, count: number) {
  if (mode === 'pdf') {
    return `I found ${count} option${count === 1 ? '' : 's'} described in the uploaded PDFs. Select one to view its details.`
  }
  if (mode === 'ai_generated') {
    return `${label ?? 'AI-generated suggestions – Not found in uploaded PDFs.'}\nThese are cautious supportive-care suggestions, not a diagnosis or a substitute for medical care. Review each option's safety information.`
  }
  return 'I could not identify a safe suggestion from the available information. Please consult a qualified health professional.'
}

function renderRemedyExplanation(content: string) {
  return content.split('\n').map((line, index) => {
    const heading = line.match(/^#{1,3}\s+(.+)$/) ?? line.match(/^\*\*(.+)\*\*$/)
    if (heading) {
      return <strong key={index} className="block pt-2 first:pt-0">{heading[1]}</strong>
    }

    const bullet = line.match(/^\s*[-*]\s+(.+)$/)
    const numbered = line.match(/^\s*(\d+)[.)]\s+(.+)$/)
    const text = bullet?.[1] ?? numbered?.[2] ?? line
    const parts = text.split(/(\*\*[^*]+\*\*)/g).map((part, partIndex) => (
      part.startsWith('**') && part.endsWith('**')
        ? <strong key={partIndex}>{part.slice(2, -2)}</strong>
        : part
    ))

    if (bullet) return <span key={index} className="block pl-3">• {parts}</span>
    if (numbered) return <span key={index} className="block pl-3">{numbered[1]}. {parts}</span>
    if (!line) return <span key={index} aria-hidden="true" className="block h-2" />
    return <span key={index} className="block">{parts}</span>
  })
}

export default function Dashboard() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<VisibleChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sending, setSending] = useState(false)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [loadingConversations, setLoadingConversations] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sendingRef = useRef(false)
  const pendingTurnRef = useRef<PendingTurn | null>(null)
  const activeSelection = selectedContext(messages)

  useEffect(() => {
    if (!user) return
    return observeConversations(user.uid, (items) => {
      setConversations(items)
      setLoadingConversations(false)
    }, (listenerError) => {
      setError(listenerError.message)
      setLoadingConversations(false)
    })
  }, [user])

  useEffect(() => {
    if (!user || !activeConversationId) return
    return observeMessages(user.uid, activeConversationId, (loadedMessages) => {
      setMessages(loadedMessages.map((message) => ({
        ...message,
        sources: message.metadata?.sources,
      })))
      setLoadingMessages(false)
    }, (listenerError) => {
      setError(listenerError.message)
      setLoadingMessages(false)
    })
  }, [user, activeConversationId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  async function handleSignOut() {
    try {
      await signOut(auth)
      navigate('/login', { replace: true })
    } catch {
      setError('Could not sign out. Please try again.')
    }
  }

  function startNewChat() {
    if (sendingRef.current) return
    pendingTurnRef.current = null
    setActiveConversationId(null)
    setMessages([])
    setLoadingMessages(false)
    setDraft('')
    setSidebarOpen(false)
    setError(null)
  }

  function openConversation(conversationId: string) {
    if (sendingRef.current) return
    pendingTurnRef.current = null
    setMessages([])
    setLoadingMessages(true)
    setError(null)
    setActiveConversationId(conversationId)
    setSidebarOpen(false)
  }

  async function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const content = draft.trim()
    if (!user || !content || sendingRef.current) return

    let pendingTurn = pendingTurnRef.current
    if (!pendingTurn || pendingTurn.content !== content || pendingTurn.baseConversationId !== activeConversationId) {
      pendingTurn = {
        content,
        baseConversationId: activeConversationId,
        conversationId: activeConversationId ?? crypto.randomUUID(),
        isNewConversation: activeConversationId === null,
        userMessageId: crypto.randomUUID(),
        assistantMessageId: crypto.randomUUID(),
      }
      pendingTurnRef.current = pendingTurn
    }

    sendingRef.current = true
    setSending(true)
    setDraft('')
    setError(null)
    setMessages((current) => current.some((message) => message.id === pendingTurn?.userMessageId)
      ? current
      : [...current, {
        id: pendingTurn!.userMessageId,
        role: 'user',
        content: pendingTurn!.content,
        timestamp: Timestamp.fromMillis(Date.now()),
      }])

    try {
      const selected = selectedContext(messages)
      let assistantContent: string
      let assistantMetadata: ChatMessageMetadata
      let assistantSources: ChatSource[]
      if (selected) {
        const response = await askRemedyFollowUp(
          selected.problem,
          selected.remedy,
          pendingTurn.content,
          priorTurns(messages),
        )
        assistantContent = response.answer ?? 'AyurBot could not answer this follow-up.'
        assistantSources = response.sources
        assistantMetadata = {
          kind: 'follow_up',
          problem: selected.problem,
          selectedRemedy: selected.remedy,
          sources: response.sources,
        }
      } else {
        const response = await recommendRemedies(pendingTurn.content)
        assistantContent = recommendationSummary(response.mode, response.label, response.remedies.length)
        assistantSources = response.sources
        assistantMetadata = {
          kind: 'recommendations',
          problem: response.problem,
          remedies: response.remedies,
          sources: response.sources,
        }
      }
      const assistantMessage: VisibleChatMessage = {
        id: pendingTurn.assistantMessageId,
        role: 'assistant',
        content: assistantContent,
        timestamp: Timestamp.now(),
        sources: assistantSources,
        metadata: assistantMetadata,
      }
      setMessages((current) => [...current, assistantMessage])

      try {
        const savedConversationId = await saveUserTurn(
          user.uid,
          pendingTurn.conversationId,
          pendingTurn.isNewConversation,
          pendingTurn.userMessageId,
          pendingTurn.assistantMessageId,
          pendingTurn.content,
          assistantMessage.content,
          assistantMetadata,
        )
        pendingTurnRef.current = null
        if (pendingTurn.isNewConversation) {
          setLoadingMessages(true)
          setActiveConversationId(savedConversationId)
        }
      } catch {
        throw new Error('The answer could not be saved to chat history. Check your connection and Firestore rules, then retry.')
      }
    } catch (requestError) {
      setMessages((current) => current.filter((message) =>
        message.id !== pendingTurn?.userMessageId && message.id !== pendingTurn?.assistantMessageId,
      ))
      setDraft(pendingTurn.content)
      setError(requestError instanceof ChatApiError
        ? requestError.message
        : requestError instanceof Error
          ? requestError.message
          : 'AyurBot is temporarily unavailable. Please try again.')
    } finally {
      sendingRef.current = false
      setSending(false)
    }
  }

  async function handleSelectRemedy(remedy: RemedyCardData, problem: string) {
    if (!user || sendingRef.current) return
    const pendingTurn: PendingTurn = {
      content: `View this remedy: ${remedy.name}`,
      baseConversationId: activeConversationId,
      conversationId: activeConversationId ?? crypto.randomUUID(),
      isNewConversation: activeConversationId === null,
      userMessageId: crypto.randomUUID(),
      assistantMessageId: crypto.randomUUID(),
    }
    sendingRef.current = true
    setSending(true)
    setError(null)
    setMessages((current) => [...current, {
      id: pendingTurn.userMessageId,
      role: 'user',
      content: pendingTurn.content,
      timestamp: Timestamp.fromMillis(Date.now()),
    }])

    try {
      const response = await explainRemedy(problem, remedy)
      const assistantContent = response.answer ?? 'AyurBot could not explain this remedy.'
      const metadata: ChatMessageMetadata = {
        kind: 'remedy_detail',
        problem,
        selectedRemedy: remedy,
        sources: response.sources,
      }
      const assistantMessage: VisibleChatMessage = {
        id: pendingTurn.assistantMessageId,
        role: 'assistant',
        content: assistantContent,
        timestamp: Timestamp.now(),
        sources: response.sources,
        metadata,
      }
      setMessages((current) => [...current, assistantMessage])
      const savedConversationId = await saveUserTurn(
        user.uid,
        pendingTurn.conversationId,
        pendingTurn.isNewConversation,
        pendingTurn.userMessageId,
        pendingTurn.assistantMessageId,
        pendingTurn.content,
        assistantContent,
        metadata,
      )
      if (pendingTurn.isNewConversation) {
        setLoadingMessages(true)
        setActiveConversationId(savedConversationId)
      }
    } catch (requestError) {
      setMessages((current) => current.filter((message) =>
        message.id !== pendingTurn.userMessageId && message.id !== pendingTurn.assistantMessageId,
      ))
      setError(requestError instanceof ChatApiError
        ? requestError.message
        : requestError instanceof Error
          ? requestError.message
          : 'AyurBot is temporarily unavailable. Please try again.')
    } finally {
      sendingRef.current = false
      setSending(false)
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <main className="flex h-[100dvh] min-h-[480px] overflow-hidden bg-[#f5f3eb] text-[#20392e]">
      {sidebarOpen && (
        <button type="button" aria-label="Close chat sidebar" onClick={() => setSidebarOpen(false)} className="fixed inset-0 z-30 bg-[#18231b]/35 md:hidden" />
      )}

      <aside className={`fixed inset-y-0 left-0 z-40 flex w-[min(82vw,300px)] flex-col border-r border-[#dce0d5] bg-[#f1f1e9] transition-transform md:static md:z-auto md:w-[280px] md:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex h-[72px] shrink-0 items-center justify-between border-b border-[#dce0d5] px-5">
          <p className="font-serif text-[25px] text-[#244d3a]">AyurBot</p>
          <button type="button" aria-label="Close chat sidebar" onClick={() => setSidebarOpen(false)} className="rounded-sm p-2 text-[#53665a] hover:bg-[#e4e8df] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#31543c] md:hidden">×</button>
        </div>

        <div className="px-4 py-4">
          <button type="button" onClick={startNewChat} disabled={sending} className="flex w-full items-center gap-2 rounded-sm border border-[#b7c2b2] bg-transparent px-3 py-2.5 text-left text-sm font-medium text-[#31543c] hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#31543c] disabled:cursor-wait disabled:opacity-60">
            <span aria-hidden="true" className="text-lg leading-none">+</span> New Chat
          </button>
        </div>

        <div className="px-5 pb-2 text-xs font-semibold uppercase tracking-[0.12em] text-[#778477]">Chats</div>
        <nav aria-label="Previous conversations" className="min-h-0 flex-1 space-y-1 overflow-y-auto px-3 pb-3">
          {loadingConversations && <p className="px-2 py-3 text-sm text-[#778477]">Loading chats...</p>}
          {!loadingConversations && conversations.length === 0 && <p className="px-2 py-3 text-sm leading-5 text-[#778477]">Your saved conversations will appear here.</p>}
          {conversations.map((conversation) => (
            <button
              key={conversation.id}
              type="button"
              onClick={() => openConversation(conversation.id)}
              disabled={sending}
              aria-current={activeConversationId === conversation.id ? 'page' : undefined}
              className={`w-full truncate rounded-sm px-2.5 py-2.5 text-left text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#31543c] ${activeConversationId === conversation.id ? 'bg-[#e2e8dd] font-medium text-[#244d3a]' : 'text-[#53665a] hover:bg-[#e9ece4]'}`}
            >
              {conversation.title}
            </button>
          ))}
        </nav>

        <div className="shrink-0 border-t border-[#dce0d5] p-4">
          <div className="mb-3 flex min-w-0 items-center gap-3">
            {user && <ProfileIcon user={user} />}
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{user?.displayName || 'AyurBot user'}</p>
              <p className="truncate text-xs text-[#778477]">{user?.email}</p>
            </div>
          </div>
          <button type="button" onClick={handleSignOut} className="w-full rounded-sm px-2 py-2 text-left text-sm text-[#53665a] hover:bg-[#e9ece4] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#31543c]">Log out</button>
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[72px] shrink-0 items-center gap-3 border-b border-[#dce0d5] px-4 sm:px-6">
          <button type="button" aria-label="Open chat sidebar" onClick={() => setSidebarOpen(true)} className="rounded-sm p-2 text-[#53665a] hover:bg-[#e9ece4] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#31543c] md:hidden">
            <svg viewBox="0 0 24 24" aria-hidden="true" className="size-5 fill-none stroke-current" strokeWidth="1.7"><path d="M4 6h16M4 12h16M4 18h16" strokeLinecap="round" /></svg>
          </button>
          <p className="font-serif text-xl text-[#31543c] md:hidden">AyurBot</p>
          <div className="hidden text-sm text-[#778477] md:block">{activeConversationId ? conversations.find((item) => item.id === activeConversationId)?.title : 'New conversation'}</div>
        </header>

        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-8 sm:px-8">
            {loadingMessages && messages.length === 0 ? (
              <div className="flex min-h-full items-center justify-center text-sm text-[#778477]">Loading conversation…</div>
            ) : messages.length === 0 ? (
              <div className="flex min-h-full items-center justify-center px-2 text-center">
                <p className="max-w-lg font-serif text-2xl leading-relaxed text-[#47634d] sm:text-[28px]">Your companion for exploring Ayurveda and home remedies.</p>
              </div>
            ) : (
              <div className="mx-auto flex w-full max-w-3xl flex-col gap-7">
                {messages.map((message) => {
                  const isUser = message.role === 'user'
                  return (
                    <article key={message.id} className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
                      <span className={`mt-0.5 grid size-9 shrink-0 place-items-center rounded-full ${isUser ? 'bg-[#e1e8dc] text-[#31543c]' : 'bg-[#31543c] text-[#f5f3eb]'}`} aria-label={isUser ? 'You' : 'AyurBot'}>
                        {isUser ? <HumanIcon /> : <RobotIcon />}
                      </span>
                      <div className={`max-w-[min(82%,640px)] ${isUser ? 'text-right' : 'text-left'}`}>
                        <p className="mb-1.5 text-xs font-semibold text-[#718071]">{isUser ? 'You' : 'AyurBot'}</p>
                        {!isUser && message.metadata?.kind === 'remedy_detail'
                          ? (
                            <div className="whitespace-pre-wrap break-words rounded-sm border border-[#e0e3da] bg-[#fbfaf6] px-4 py-3 text-left text-sm leading-6 text-[#405244]">
                              {renderRemedyExplanation(message.content)}
                            </div>
                          )
                          : (
                            <p className={`whitespace-pre-wrap break-words rounded-sm px-4 py-3 text-left text-sm leading-6 ${isUser ? 'bg-[#e4e9df] text-[#20392e]' : 'border border-[#e0e3da] bg-[#fbfaf6] text-[#405244]'}`}>
                              {message.content}
                            </p>
                          )}
                        {!isUser && message.metadata?.kind !== 'remedy_detail' && message.sources && message.sources.length > 0 && (
                          <div className="mt-2 text-left text-xs text-[#718071]">
                            <p className="mb-1 font-semibold">Sources</p>
                            <ul className="space-y-1">
                              {message.sources.map((source) => (
                                <li key={source.chunk_id} className="break-words">
                                  {source.source}, page {source.page}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {!isUser && message.metadata?.remedies && message.metadata.remedies.length > 0 && (
                          <div className="mt-3 text-left">
                            {message.content.startsWith('AI-generated suggestions – Not found in uploaded PDFs.')
                              && <p className="mb-2 text-xs font-semibold text-[#793d2e]">AI-generated suggestions – Not found in uploaded PDFs.</p>}
                            {message.metadata.remedies.map((remedy) => (
                              <RemedyCard
                                key={remedy.id}
                                remedy={remedy}
                                selected={activeSelection?.remedy.id === remedy.id}
                                disabled={sending}
                                onSelect={() => void handleSelectRemedy(remedy, message.metadata!.problem)}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    </article>
                  )
                })}
                {sending && <p role="status" aria-live="polite" className="text-sm text-[#778477]">AyurBot is reviewing relevant information…</p>}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          <div className="shrink-0 border-t border-[#dce0d5] px-4 py-4 sm:px-8 sm:py-5">
            <div className="mx-auto max-w-3xl">
              {activeSelection && (
                <p className="mb-2 text-xs text-[#718071]">
                  Follow-up context: <strong>{activeSelection.remedy.name}</strong> for {activeSelection.problem}
                </p>
              )}
              {error && <p role="alert" className="mb-3 rounded-sm border-l-2 border-[#a75f46] bg-[#f8eae3] px-3 py-2 text-sm text-[#793d2e]">{error}</p>}
              <form onSubmit={handleSend} className="flex items-end gap-2 rounded-sm border border-[#cbd3c7] bg-[#fffefa] p-2 focus-within:border-[#81927e]">
                <label htmlFor="chat-draft" className="sr-only">Message</label>
                <textarea
                  id="chat-draft"
                  value={draft}
                  disabled={sending}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  rows={1}
                  placeholder="Ask about Ayurveda and home remedies..."
                  className="max-h-40 min-h-10 min-w-0 flex-1 resize-y bg-transparent px-2 py-2 text-sm leading-5 text-[#20392e] outline-none placeholder:text-[#889286]"
                />
                <button type="submit" disabled={!draft.trim() || sending} aria-label="Send message" className="grid size-10 shrink-0 place-items-center rounded-sm bg-[#31543c] text-white hover:bg-[#24452f] disabled:cursor-not-allowed disabled:opacity-45 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#31543c]">
                  {sending ? <span className="text-xs">...</span> : <svg viewBox="0 0 24 24" aria-hidden="true" className="size-5 fill-none stroke-current" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M6 11l6-6 6 6" /></svg>}
                </button>
              </form>
              <p className="mt-2 text-center text-[11px] text-[#849083]">Enter to send · Shift + Enter for a new line</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}