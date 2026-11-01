import {
  collection,
  doc,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  Timestamp,
  writeBatch,
  type DocumentData,
  type Unsubscribe,
} from 'firebase/firestore'
import { getFirestore } from 'firebase/firestore'
import { firebaseApp } from './firebase'
import type { ChatSource, RemedyCard } from './chatApi'

const db = getFirestore(firebaseApp)

export type Conversation = {
  id: string
  title: string
  createdAt: Timestamp | null
  updatedAt: Timestamp | null
}

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Timestamp
  metadata?: ChatMessageMetadata
}

export type ChatMessageMetadata = {
  kind: 'recommendations' | 'remedy_detail' | 'follow_up'
  problem: string
  remedies?: RemedyCard[]
  selectedRemedy?: RemedyCard
  sources?: ChatSource[]
}

function conversationCollection(userId: string) {
  return collection(db, 'users', userId, 'conversations')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseSource(value: unknown): ChatSource | null {
  if (!isRecord(value)
    || typeof value.source !== 'string'
    || typeof value.page !== 'number'
    || typeof value.chunk_id !== 'string') return null
  return {
    source: value.source,
    page: value.page,
    chunk_id: value.chunk_id,
    ...(typeof value.snippet === 'string' ? { snippet: value.snippet } : {}),
  }
}

function parseRemedy(value: unknown): RemedyCard | null {
  if (!isRecord(value)
    || typeof value.id !== 'string'
    || (value.kind !== 'pdf' && value.kind !== 'ai_generated')
    || typeof value.name !== 'string'
    || typeof value.description !== 'string') return null
  const sources = Array.isArray(value.sources)
    ? value.sources.map(parseSource).filter((source): source is ChatSource => source !== null)
    : []
  const optionalText = (key: string) => typeof value[key] === 'string' ? value[key] as string : null
  return {
    id: value.id,
    kind: value.kind,
    name: value.name,
    description: value.description,
    ingredients: optionalText('ingredients'),
    preparation: optionalText('preparation'),
    traditional_or_supportive_basis: optionalText('traditional_or_supportive_basis'),
    evidence: optionalText('evidence'),
    possible_benefits: optionalText('possible_benefits'),
    side_effects: optionalText('side_effects'),
    who_should_avoid: optionalText('who_should_avoid'),
    precautions_interactions: optionalText('precautions_interactions'),
    when_to_seek_care: optionalText('when_to_seek_care'),
    sources,
  }
}

function parseMessageMetadata(value: unknown): ChatMessageMetadata | undefined {
  if (!isRecord(value)
    || !['recommendations', 'remedy_detail', 'follow_up'].includes(String(value.kind))
    || typeof value.problem !== 'string') return undefined
  const remedies = Array.isArray(value.remedies)
    ? value.remedies.map(parseRemedy).filter((remedy): remedy is RemedyCard => remedy !== null)
    : undefined
  const selectedRemedy = parseRemedy(value.selectedRemedy)
  const sources = Array.isArray(value.sources)
    ? value.sources.map(parseSource).filter((source): source is ChatSource => source !== null)
    : undefined
  return {
    kind: value.kind as ChatMessageMetadata['kind'],
    problem: value.problem,
    ...(remedies ? { remedies } : {}),
    ...(selectedRemedy ? { selectedRemedy } : {}),
    ...(sources ? { sources } : {}),
  }
}

export function observeConversations(
  userId: string,
  onChange: (conversations: Conversation[]) => void,
  onError: (error: Error) => void,
): Unsubscribe {
  const conversationsQuery = query(conversationCollection(userId), orderBy('updatedAt', 'desc'))
  return onSnapshot(conversationsQuery, (snapshot) => {
    onChange(snapshot.docs.map((item) => {
      const data = item.data() as DocumentData
      return {
        id: item.id,
        title: typeof data.title === 'string' ? data.title : 'New conversation',
        createdAt: data.createdAt instanceof Timestamp ? data.createdAt : null,
        updatedAt: data.updatedAt instanceof Timestamp ? data.updatedAt : null,
      }
    }))
  }, onError)
}

export function observeMessages(
  userId: string,
  conversationId: string,
  onChange: (messages: ChatMessage[]) => void,
  onError: (error: Error) => void,
): Unsubscribe {
  const messagesQuery = query(
    collection(db, 'users', userId, 'conversations', conversationId, 'messages'),
    orderBy('timestamp', 'asc'),
  )
  return onSnapshot(messagesQuery, (snapshot) => {
    onChange(snapshot.docs.map((item) => {
      const data = item.data() as DocumentData
      return {
        id: item.id,
        role: data.role === 'assistant' ? 'assistant' : 'user',
        content: typeof data.content === 'string' ? data.content : '',
        timestamp: data.timestamp instanceof Timestamp ? data.timestamp : Timestamp.now(),
        metadata: parseMessageMetadata(data.metadata),
      }
    }))
  }, onError)
}

function makeTitle(message: string) {
  const normalized = message.replace(/\s+/g, ' ').trim()
  return normalized.length > 52 ? `${normalized.slice(0, 49).trimEnd()}...` : normalized
}

export async function saveUserTurn(
  userId: string,
  conversationId: string,
  isNewConversation: boolean,
  messageId: string,
  assistantMessageId: string,
  userContent: string,
  assistantContent: string,
  assistantMetadata?: ChatMessageMetadata,
): Promise<string> {
  const batch = writeBatch(db)
  const conversationRef = doc(db, 'users', userId, 'conversations', conversationId)
  const messageCollection = collection(conversationRef, 'messages')
  const userMessageRef = doc(messageCollection, messageId)
  const assistantMessageRef = doc(messageCollection, assistantMessageId)
  const now = Date.now()

  if (isNewConversation) {
    batch.set(conversationRef, {
      title: makeTitle(userContent),
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    })
  } else {
    batch.update(conversationRef, { updatedAt: serverTimestamp() })
  }

  batch.set(userMessageRef, {
    role: 'user',
    content: userContent,
    timestamp: Timestamp.fromMillis(now),
  })
  batch.set(assistantMessageRef, {
    role: 'assistant',
    content: assistantContent,
    timestamp: Timestamp.fromMillis(now + 1),
    ...(assistantMetadata ? { metadata: assistantMetadata } : {}),
  })

  await batch.commit()
  return conversationId
}