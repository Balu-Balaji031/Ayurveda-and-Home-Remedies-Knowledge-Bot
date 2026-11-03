export type ChatSource = {
  source: string
  page: number
  chunk_id: string
  snippet?: string
}

export type ChatResponse = {
  answer: string | null
  sources: ChatSource[]
  error: string | null
}

export type RemedyCard = {
  id: string
  kind: 'pdf' | 'ai_generated'
  name: string
  description: string
  ingredients: string | null
  preparation: string | null
  traditional_or_supportive_basis?: string | null
  evidence?: string | null
  possible_benefits?: string | null
  side_effects?: string | null
  who_should_avoid?: string | null
  precautions_interactions?: string | null
  when_to_seek_care?: string | null
  sources: ChatSource[]
}

export type RecommendationResponse = {
  problem: string
  mode: 'pdf' | 'ai_generated' | 'abstain'
  label: string | null
  remedies: RemedyCard[]
  sources: ChatSource[]
  error: string | null
}

export type FollowUpTurn = {
  question: string
  answer: string
}

export type RemedyTextResponse = {
  answer: string | null
  sources: ChatSource[]
  error: string | null
}

export class ChatApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ChatApiError'
    this.status = status
  }
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'https://ayurveda-and-home-remedies-knowledge-bot.onrender.com').replace(/\/+$/, '')
export const MAX_CHAT_QUESTION_LENGTH = 2_000

async function postWorkflow<T>(path: string, body: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(105_000),
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new ChatApiError('AyurBot is taking longer than expected to respond. Please try again.', 0)
    }
    throw new ChatApiError(
      'Could not reach the AyurBot backend. Check that FastAPI is running at the configured API URL and that CORS allows this site.',
      0,
    )
  }

  let payload: T & { error?: unknown }
  try {
    payload = await response.json() as T & { error?: unknown }
  } catch {
    throw new ChatApiError('AyurBot returned an unreadable response. Please try again.', response.status)
  }
  if (!response.ok) {
    throw new ChatApiError(
      typeof payload.error === 'string' && payload.error
        ? payload.error
        : 'AyurBot could not answer right now. Please try again.',
      response.status,
    )
  }
  return payload
}

export async function recommendRemedies(problem: string): Promise<RecommendationResponse> {
  const normalizedProblem = problem.trim()
  if (!normalizedProblem) throw new ChatApiError('Describe the health problem before continuing.', 422)
  if (normalizedProblem.length > MAX_CHAT_QUESTION_LENGTH) {
    throw new ChatApiError(`Questions must be ${MAX_CHAT_QUESTION_LENGTH} characters or fewer.`, 422)
  }
  return postWorkflow<RecommendationResponse>('/api/recommendations', { problem: normalizedProblem })
}

export async function explainRemedy(problem: string, remedy: RemedyCard): Promise<RemedyTextResponse> {
  return postWorkflow<RemedyTextResponse>('/api/remedies/explain', { problem, remedy })
}

export async function askRemedyFollowUp(
  problem: string,
  remedy: RemedyCard,
  question: string,
  history: FollowUpTurn[],
): Promise<RemedyTextResponse> {
  const normalizedQuestion = question.trim()
  if (!normalizedQuestion) throw new ChatApiError('Enter a follow-up question before sending.', 422)
  if (normalizedQuestion.length > MAX_CHAT_QUESTION_LENGTH) {
    throw new ChatApiError(`Questions must be ${MAX_CHAT_QUESTION_LENGTH} characters or fewer.`, 422)
  }
  return postWorkflow<RemedyTextResponse>('/api/chat/follow-up', {
    problem,
    remedy,
    question: normalizedQuestion,
    history: history.slice(-8),
  })
}

export async function askAyurBot(question: string): Promise<ChatResponse> {
  const normalizedQuestion = question.trim()
  if (!normalizedQuestion) throw new ChatApiError('Enter a question before sending.', 422)
  if (normalizedQuestion.length > MAX_CHAT_QUESTION_LENGTH) {
    throw new ChatApiError(`Questions must be ${MAX_CHAT_QUESTION_LENGTH} characters or fewer.`, 422)
  }

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: normalizedQuestion }),
      signal: AbortSignal.timeout(105_000),
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new ChatApiError('AyurBot is taking longer than expected to respond. Please try again.', 0)
    }
    throw new ChatApiError(
      'Could not reach the AyurBot backend. Check that FastAPI is running at the configured API URL and that CORS allows this site.',
      0,
    )
  }

  let payload: ChatResponse
  try {
    payload = await response.json() as ChatResponse
  } catch {
    throw new ChatApiError('AyurBot returned an unreadable response. Please try again.', response.status)
  }

  if (!response.ok) {
    throw new ChatApiError(
      typeof payload.error === 'string' && payload.error
        ? payload.error
        : 'AyurBot could not answer right now. Please try again.',
      response.status,
    )
  }
  if (typeof payload.answer !== 'string' || !payload.answer.trim()) {
    throw new ChatApiError('AyurBot returned an empty answer. Please try again.', response.status)
  }

  return {
    answer: payload.answer,
    sources: Array.isArray(payload.sources) ? payload.sources : [],
    error: null,
  }
}