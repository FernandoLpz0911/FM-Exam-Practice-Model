import { useCallback, useEffect, useRef, useState } from 'react'
import katex from 'katex'
import { api } from '../api'
import type { MockExamProblem, MockExamResult } from '../types'
import './MockExam.css'

const EXAM_SECONDS = 3 * 60 * 60  // 3 hours

/** Render KaTeX math in text. Display ($$) runs before inline ($) to avoid partial matches. */
function renderMath(text: string): string {
  let out = text.replace(/\$\$([^$]+)\$\$/g, (_m, expr: string) => {
    try { return katex.renderToString(expr, { throwOnError: false, displayMode: true }) }
    catch { return expr }
  })
  out = out.replace(/\$([^$\n]+)\$/g, (_m, expr: string) => {
    try { return katex.renderToString(expr, { throwOnError: false }) }
    catch { return expr }
  })
  return out
}

function fmtTime(s: number): string {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

type Phase = 'idle' | 'loading' | 'exam' | 'grading' | 'results' | 'error'

export default function MockExam() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [nQuestions, setNQuestions] = useState(30)
  const [problems, setProblems] = useState<MockExamProblem[]>([])
  const [sessionId, setSessionId] = useState<number | null>(null)
  const [answers, setAnswers] = useState<Record<number, string>>({})  // item_id → choice
  const [flagged, setFlagged] = useState<Set<number>>(new Set())
  const [currentIdx, setCurrentIdx] = useState(0)
  const [timeLeft, setTimeLeft] = useState(EXAM_SECONDS)
  const [result, setResult] = useState<MockExamResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const startTime = useRef<number>(Date.now())
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current)
  }

  const startExam = useCallback(async () => {
    setPhase('loading')
    setAnswers({})
    setFlagged(new Set())
    setCurrentIdx(0)
    setTimeLeft(EXAM_SECONDS)
    setResult(null)
    try {
      const data = await api.startMockExam(nQuestions)
      setSessionId(data.session_id)
      setProblems(data.problems)
      startTime.current = Date.now()
      setPhase('exam')
      timerRef.current = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) {
            stopTimer()
            return 0
          }
          return prev - 1
        })
      }, 1000)
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }, [nQuestions])

  const submitExam = useCallback(async () => {
    if (sessionId === null) return
    stopTimer()
    setPhase('grading')
    const elapsed = Math.floor((Date.now() - startTime.current) / 1000)
    const answerList = problems.map(p => ({
      item_id: p.item_id,
      user_answer: answers[p.item_id] ?? '',
    }))
    try {
      const res = await api.gradeMockExam(sessionId, answerList, elapsed)
      setResult(res)
      setPhase('results')
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }, [sessionId, problems, answers])

  // Auto-submit when time runs out
  useEffect(() => {
    if (timeLeft === 0 && phase === 'exam') {
      submitExam()
    }
  }, [timeLeft, phase, submitExam])

  useEffect(() => () => stopTimer(), [])

  const toggleFlag = (itemId: number) => {
    setFlagged(prev => {
      const next = new Set(prev)
      if (next.has(itemId)) next.delete(itemId)
      else next.add(itemId)
      return next
    })
  }

  if (phase === 'idle') {
    return (
      <div className="me-idle">
        <h2>Mock Exam</h2>
        <p className="me-desc">
          Simulates the SOA Exam FM format: timed, no immediate feedback,
          all results shown at the end.
        </p>
        <div className="me-config">
          <label className="me-label">
            Questions:
            <select
              className="me-select"
              value={nQuestions}
              onChange={e => setNQuestions(Number(e.target.value))}
            >
              <option value={10}>10 (quick drill)</option>
              <option value={20}>20 (half-length)</option>
              <option value={30}>30 (full exam)</option>
            </select>
          </label>
          <p className="me-hint">Time limit: 3:00:00 (SOA standard)</p>
        </div>
        <button className="btn-primary me-start" onClick={startExam}>
          Start Exam
        </button>
      </div>
    )
  }

  if (phase === 'loading') {
    return <div className="me-loading">Generating exam…</div>
  }

  if (phase === 'grading') {
    return <div className="me-loading">Grading…</div>
  }

  if (phase === 'error') {
    return (
      <div className="me-error">
        <strong>Error</strong>
        <p>{error}</p>
      </div>
    )
  }

  if (phase === 'results' && result) {
    const pct = Math.round(result.score * 100)
    const pass = result.score >= 0.70
    return (
      <div className="me-results">
        <div className={`me-score-banner ${pass ? 'pass' : 'fail'}`}>
          <span className="me-score-big">{pct}%</span>
          <span className="me-score-label">
            {result.n_correct} / {result.n_total} correct
            &nbsp;·&nbsp;
            {fmtTime(result.elapsed_s)} elapsed
            &nbsp;·&nbsp;
            {pass ? '≥ 70% — PASS' : '< 70% — keep studying'}
          </span>
        </div>

        <div className="me-cat-grid">
          {Object.entries(result.by_category).map(([cat, v]) => (
            <div key={cat} className="me-cat-card">
              <div className="me-cat-name">{cat}</div>
              <div className="me-cat-score">
                {v.correct}/{v.total}
                &nbsp;·&nbsp;
                {Math.round(v.correct / v.total * 100)}%
              </div>
              <div className="me-cat-bar">
                <div
                  className="me-cat-fill"
                  style={{
                    width: `${Math.round(v.correct / v.total * 100)}%`,
                    background: v.correct / v.total >= 0.7 ? '#4ade80' : '#f87171',
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="me-detail">
          <h3>Question review</h3>
          <div className="me-review-list">
            {result.results.map((r, i) => {
              const prob = problems.find(p => p.item_id === r.item_id)
              return (
                <div key={r.item_id} className={`me-review-row ${r.is_correct ? 'correct' : 'wrong'}`}>
                  <span className="me-q-num">Q{i + 1}</span>
                  <span className="me-icon">{r.is_correct ? '✓' : '✗'}</span>
                  <span className="me-concept-name">{prob?.concept_name}</span>
                  {!r.is_correct && (
                    <span className="me-answers">
                      Your: <em>{r.user_answer || '—'}</em>
                      &nbsp;Correct: <strong>{r.correct_answer}</strong>
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        <button className="btn-primary me-again" onClick={() => setPhase('idle')}>
          Take Another Exam
        </button>
      </div>
    )
  }

  const prob = problems[currentIdx]
  const answered = Object.keys(answers).length
  const timerClass = timeLeft < 600 ? 'me-timer danger' : timeLeft < 1800 ? 'me-timer warn' : 'me-timer'

  return (
    <div className="me-exam">
      {/* Header bar */}
      <div className="me-exam-header">
        <span className="me-progress">
          Q{currentIdx + 1} / {problems.length}
          &nbsp;·&nbsp;
          {answered} answered
        </span>
        <span className={timerClass}>{fmtTime(timeLeft)}</span>
        <button className="btn-submit-exam" onClick={submitExam}>
          Submit Exam
        </button>
      </div>

      {/* Question navigator dots */}
      <div className="me-nav-dots">
        {problems.map((p, i) => (
          <button
            key={p.item_id}
            className={[
              'me-dot',
              i === currentIdx ? 'current' : '',
              answers[p.item_id] ? 'done' : '',
              flagged.has(p.item_id) ? 'flagged' : '',
            ].filter(Boolean).join(' ')}
            onClick={() => setCurrentIdx(i)}
            title={`Q${i + 1}: ${p.concept_name}`}
          />
        ))}
      </div>

      {/* Problem */}
      <div className="me-problem">
        <div className="me-problem-meta">
          <span className="me-concept-tag">{prob.concept_name}</span>
          <span className="me-cat-tag">{prob.category}</span>
          <button
            className={`me-flag-btn ${flagged.has(prob.item_id) ? 'active' : ''}`}
            onClick={() => toggleFlag(prob.item_id)}
          >
            {flagged.has(prob.item_id) ? '🚩 Flagged' : '⚑ Flag'}
          </button>
        </div>

        <div
          className="me-statement"
          dangerouslySetInnerHTML={{ __html: renderMath(prob.statement) }}
        />

        <div className="me-choices">
          {prob.choices.map(c => (
            <button
              key={c}
              className={`me-choice ${answers[prob.item_id] === c ? 'selected' : ''}`}
              onClick={() => setAnswers(prev => ({ ...prev, [prob.item_id]: c }))}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* Prev / Next */}
      <div className="me-nav-btns">
        <button
          className="btn-nav"
          onClick={() => setCurrentIdx(i => Math.max(0, i - 1))}
          disabled={currentIdx === 0}
        >
          ← Previous
        </button>
        <button
          className="btn-nav"
          onClick={() => setCurrentIdx(i => Math.min(problems.length - 1, i + 1))}
          disabled={currentIdx === problems.length - 1}
        >
          Next →
        </button>
      </div>
    </div>
  )
}
