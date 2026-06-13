/** HTTP client for the FastAPI engine. All calls route through /api (Vite proxy). */
import type { AnswerOut, CardState, ConceptDetail, DKTStatus, JournalStats, MockExamResult, NextOut, ReadinessOut, SampleExamIn, TimingEntry, TrainOut } from './types'

const BASE = '/api'

async function post<T>(path: string, body?: unknown): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!resp.ok) throw new Error(`POST ${path} → ${resp.status}`)
  return resp.json() as Promise<T>
}

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE}${path}`)
  if (!resp.ok) throw new Error(`GET ${path} → ${resp.status}`)
  return resp.json() as Promise<T>
}

export const api = {
  createSession: () => post<{ session_id: number; started_at: string }>('/session'),

  closeSession: (sessionId: number) =>
    post<{ ended_at: string }>(`/session/${sessionId}/close`),

  getNext: (sessionId: number) =>
    get<NextOut>(`/next?session_id=${sessionId}`),

  postAnswer: (itemId: number, userAnswer: string, elapsedMs: number) =>
    post<AnswerOut>('/answer', { item_id: itemId, user_answer: userAnswer, elapsed_ms: elapsedMs }),

  getState: () =>
    get<{ states: CardState[] }>('/state'),

  getDKTStatus: () =>
    get<DKTStatus>('/dkt/status'),

  train: () =>
    post<TrainOut>('/train'),

  getReadiness: (snapshot = false) =>
    get<ReadinessOut>(`/readiness?snapshot=${snapshot}`),

  getReadinessHistory: () =>
    get<{ snapshots: { id: number; taken_at: string; score: number }[] }>('/readiness/history'),

  getJournal: () =>
    get<{ stats: JournalStats; sample_exams: Record<string, unknown>[]; recent_readiness: { score: number; taken_at: string }[] }>('/journal'),

  postSampleExam: (body: SampleExamIn) =>
    post<{ id: number; predicted: number | null }>('/sample-exam', body),

  getConcept: (conceptId: string) =>
    get<ConceptDetail>(`/concept/${conceptId}`),

  startMockExam: (n = 30) =>
    post<{ session_id: number; problems: import('./types').MockExamProblem[] }>(
      `/mock-exam/start?n=${n}`
    ),

  gradeMockExam: (
    sessionId: number,
    answers: { item_id: number; user_answer: string }[],
    elapsedS: number,
  ) =>
    post<MockExamResult>('/mock-exam/grade', {
      session_id: sessionId,
      answers,
      elapsed_s: elapsedS,
    }),

  getTiming: () =>
    get<TimingEntry[]>('/analytics/timing'),

  getHint: (itemId: number) =>
    get<{ hint_steps: string[]; total: number }>(`/hint/${itemId}`),

  getSolution: (itemId: number) =>
    get<{ steps: string[]; correct_answer: string | null }>(`/solution/${itemId}`),

  chartUrl: (kind: 'timeline' | 'category' | 'predicted-vs-actual') =>
    `${BASE}/readiness/chart/${kind}`,
}
