import { useCallback, useEffect, useRef, useState } from 'react'
import katex from 'katex'
import { api } from '../api'
import type { AnswerOut, NextOut } from '../types'
import './StudySession.css'

/** Render KaTeX math in text. Display ($$) runs before inline ($) to avoid partial matches. */
function renderMath(text: string): string {
  let out = text.replace(/\$\$([^$]+)\$\$/g, (_m, expr: string) => {
    try {
      return katex.renderToString(expr, { throwOnError: false, displayMode: true })
    } catch { return expr }
  })
  // inline math $...$
  out = out.replace(/\$([^$\n]+)\$/g, (_m, expr: string) => {
    try {
      return katex.renderToString(expr, { throwOnError: false })
    } catch { return expr }
  })
  return out
}

/**
 * Hand-rolled Markdown → HTML converter. A dedicated lib is avoided so KaTeX
 * can run on each cell/line before the HTML is assembled.
 */
function renderTheory(md: string): string {
  const lines = md.split('\n')
  const html: string[] = []
  let inTable = false

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i]
    const line = raw.trim()

    if (line === '') {
      if (inTable) { html.push('</tbody></table>'); inTable = false }
      html.push('<br/>')
      continue
    }

    // headings ## and ###
    if (line.startsWith('### ')) {
      html.push(`<h4 class="theory-h theory-h4">${inlineMd(line.slice(4))}</h4>`)
      continue
    }
    if (line.startsWith('## ')) {
      html.push(`<h3 class="theory-h">${inlineMd(line.slice(3))}</h3>`)
      continue
    }

    // table row |...|
    if (line.startsWith('|')) {
      const cells = line.split('|').slice(1, -1).map(c => c.trim())
      // separator row (---|---)
      if (cells.every(c => /^[-: ]+$/.test(c))) {
        if (!inTable) { html.push('<table class="theory-table"><tbody>') ; inTable = true }
        continue
      }
      if (!inTable) { html.push('<table class="theory-table"><tbody>'); inTable = true }
      const tag = (i === 0 || lines[i - 1].trim() === '') ? 'th' : 'td'
      html.push(`<tr>${cells.map(c => `<${tag}>${inlineMd(c)}</${tag}>`).join('')}</tr>`)
      continue
    }

    if (inTable) { html.push('</tbody></table>'); inTable = false }

    // list item -
    if (line.startsWith('- ')) {
      html.push(`<li>${inlineMd(line.slice(2))}</li>`)
      continue
    }

    html.push(`<p>${inlineMd(line)}</p>`)
  }

  if (inTable) html.push('</tbody></table>')
  return html.join('')
}

/** Apply math then bold/code. Order matters: bold regex would corrupt KaTeX output if run first. */
function inlineMd(text: string): string {
  let out = renderMath(text)
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>')
  return out
}

type Phase = 'loading' | 'cold_attempt' | 'intro' | 'question' | 'answered' | 'done' | 'error'

export default function StudySession() {
  const [sessionId, setSessionId] = useState<number | null>(null)
  const [phase, setPhase] = useState<Phase>('loading')
  const [problem, setProblem] = useState<NextOut | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [feedback, setFeedback] = useState<AnswerOut | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [theoryMd, setTheoryMd] = useState<string | null>(null)
  const [coldSelected, setColdSelected] = useState<string | null>(null)
  const [coldFeedback, setColdFeedback] = useState<AnswerOut | null>(null)
  const [elapsedS, setElapsedS] = useState(0)
  const [coldSolutionSteps, setColdSolutionSteps] = useState<string[]>([])
  const [coldCorrectAnswer, setColdCorrectAnswer] = useState<string | null>(null)
  const [hintSteps, setHintSteps] = useState<string[]>([])
  const [hintsShown, setHintsShown] = useState(0)
  const shownAt = useRef<number>(Date.now())
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const sessionClosedRef = useRef(false)

  const doCloseSession = useCallback((sid: number) => {
    if (sessionClosedRef.current) return
    sessionClosedRef.current = true
    api.closeSession(sid).catch(() => {})
  }, [])

  const loadNext = useCallback(async (sid: number) => {
    setPhase('loading')
    setSelected(null)
    setFeedback(null)
    setTheoryMd(null)
    setColdFeedback(null)
    setColdSolutionSteps([])
    setColdCorrectAnswer(null)
    setHintSteps([])
    setHintsShown(0)
    try {
      const next = await api.getNext(sid)
      if (next.item_id === null) {
        setPhase('done')
        setProblem(null)
      } else {
        setProblem(next)
        shownAt.current = Date.now()
        if (next.is_new_concept && next.concept_id) {
          // Fetch theory non-blocking — cold_attempt shows the problem first
          api.getConcept(next.concept_id)
            .then(concept => setTheoryMd(concept.theory_md ?? null))
            .catch(() => {})
          setColdSelected(null)
          setPhase('cold_attempt')
        } else {
          setPhase('question')
          if (next.concept_id) {
            api.getConcept(next.concept_id)
              .then(concept => setTheoryMd(concept.theory_md))
              .catch(() => {})
          }
        }
      }
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }, [])

  useEffect(() => {
    api.createSession()
      .then(({ session_id }) => {
        setSessionId(session_id)
        return loadNext(session_id)
      })
      .catch((e) => {
        setError(String(e))
        setPhase('error')
      })
  }, [loadNext])

  const handleChoice = (choice: string) => {
    if (phase !== 'question') return
    setSelected(choice)
  }

  const handleSubmit = useCallback(async () => {
    if (!problem?.item_id || selected === null || sessionId === null) return
    const elapsed = Date.now() - shownAt.current
    try {
      const result = await api.postAnswer(problem.item_id, selected, elapsed)
      setFeedback(result)
      setPhase('answered')
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }, [problem, selected, sessionId])

  const handleColdSubmit = useCallback(async () => {
    if (!problem?.item_id || sessionId === null) return
    const elapsed = Date.now() - shownAt.current
    if (coldSelected !== null) {
      try {
        const result = await api.postAnswer(problem.item_id, coldSelected, elapsed)
        setColdFeedback(result)
        setColdSolutionSteps(result.solution_steps)
        setColdCorrectAnswer(result.correct_answer)
      } catch {
        // non-fatal: still show theory even if POST fails
      }
    } else {
      // skipped — fetch full solution without grading
      try {
        const sol = await api.getSolution(problem.item_id)
        setColdSolutionSteps(sol.steps)
        setColdCorrectAnswer(sol.correct_answer)
      } catch { /* no solution available */ }
    }
    setPhase('intro')
  }, [problem, coldSelected, sessionId])

  const handleHint = useCallback(async () => {
    if (!problem?.item_id) return
    if (hintSteps.length === 0) {
      try {
        const result = await api.getHint(problem.item_id)
        setHintSteps(result.hint_steps)
        setHintsShown(1)
      } catch { /* no hint available */ }
    } else {
      setHintsShown(n => Math.min(n + 1, hintSteps.length))
    }
  }, [problem, hintSteps])

  const handleNext = useCallback(() => {
    if (sessionId !== null) loadNext(sessionId)
  }, [sessionId, loadNext])

  // Close session when all items are done
  useEffect(() => {
    if (phase === 'done' && sessionId !== null) doCloseSession(sessionId)
  }, [phase, sessionId, doCloseSession])

  // Close session if user navigates away mid-session
  useEffect(() => {
    return () => { if (sessionId !== null) doCloseSession(sessionId) }
  }, [sessionId, doCloseSession])

  // Timer: start counting when question becomes active, reset shownAt for accurate elapsed_ms
  useEffect(() => {
    if (phase === 'question') {
      shownAt.current = Date.now()
      setElapsedS(0)
      timerRef.current = setInterval(() => setElapsedS(s => s + 1), 1000)
    } else {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    }
    return () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null } }
  }, [phase])

  // Keyboard shortcuts: A/B/C/D to pick a choice, Enter to submit or advance
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (phase === 'cold_attempt' && problem?.choices) {
        const idx = ['a', 'b', 'c', 'd'].indexOf(e.key.toLowerCase())
        if (idx !== -1 && idx < problem.choices.length) {
          setColdSelected(problem.choices[idx])
          return
        }
        if (e.key === 'Enter') { void handleColdSubmit(); return }
      }
      if (phase === 'question' && problem?.choices) {
        const idx = ['a', 'b', 'c', 'd'].indexOf(e.key.toLowerCase())
        if (idx !== -1 && idx < problem.choices.length) {
          setSelected(problem.choices[idx])
          return
        }
        if (e.key === 'Enter' && selected !== null) {
          void handleSubmit()
          return
        }
      }
      if (phase === 'answered' && e.key === 'Enter') handleNext()
      if (phase === 'intro' && e.key === 'Enter' && sessionId !== null) void loadNext(sessionId)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [phase, problem, selected, sessionId, handleSubmit, handleNext, handleColdSubmit, loadNext])

  if (phase === 'error') {
    return (
      <div className="ss-error">
        <strong>Error connecting to engine.</strong>
        <p>{error}</p>
        <p className="ss-hint">
          Make sure <code>uvicorn engine.main:app --port 8001</code> is running on port 8001.
        </p>
      </div>
    )
  }

  if (phase === 'loading') {
    return <div className="ss-loading">Loading…</div>
  }

  if (phase === 'cold_attempt') {
    return (
      <div className="ss-wrap">
        <div className="ss-cold-banner">
          <div className="ss-cold-label">Cold attempt — new concept</div>
          <h2 className="ss-cold-title">{problem?.concept_name}</h2>
          <p className="ss-cold-sub">Try the problem before seeing the theory. Doesn't count yet.</p>
        </div>
        <div
          className="ss-statement"
          dangerouslySetInnerHTML={{ __html: renderMath(problem?.statement ?? '') }}
        />
        {problem?.choices && (
          <div className="ss-choices">
            {problem.choices.map(c => (
              <button
                key={c}
                className={coldSelected === c ? 'btn-choice selected' : 'btn-choice'}
                onClick={() => setColdSelected(c)}
              >
                {c}
              </button>
            ))}
          </div>
        )}
        <button
          className="btn-primary ss-intro-btn"
          onClick={() => { void handleColdSubmit() }}
        >
          {coldSelected ? 'Submit & see theory' : 'Skip to theory'} →{' '}
          <span className="ss-kbd-hint">Enter</span>
        </button>
      </div>
    )
  }

  if (phase === 'intro') {
    const gotItRight = coldFeedback?.is_correct === true
    return (
      <div className="ss-wrap">
        <div className="ss-intro-banner">
          <div className="ss-intro-label">New concept</div>
          <h2 className="ss-intro-title">{problem?.concept_name}</h2>
          {coldFeedback
            ? <p className={`ss-intro-sub ${gotItRight ? 'ss-intro-correct' : 'ss-intro-wrong'}`}>
                {gotItRight
                  ? `✓ Correct on first try — answer: ${coldFeedback.correct_answer}`
                  : `✗ Wrong — correct answer: ${coldFeedback.correct_answer}`}
              </p>
            : <p className="ss-intro-sub">Read through the material below, then start practicing.</p>
          }
        </div>
        {theoryMd
          ? <details className="ss-intro-theory-details" open={!gotItRight}>
              <summary className="ss-intro-theory-summary">
                {gotItRight ? '📖 Theory (you got it — optional review)' : '📖 Theory — review before practicing'}
              </summary>
              <div dangerouslySetInnerHTML={{ __html: renderTheory(theoryMd) }} />
            </details>
          : <div className="ss-loading">Loading theory…</div>
        }
        {coldSolutionSteps.length > 0 && (
          <details className="ss-solution" open={!gotItRight}>
            <summary>
              Worked solution{coldCorrectAnswer ? ` — answer: ${coldCorrectAnswer}` : ''} ({coldSolutionSteps.length} steps)
            </summary>
            <div
              className="ss-solution-question"
              dangerouslySetInnerHTML={{ __html: renderMath(problem?.statement ?? '') }}
            />
            <ol className="ss-steps">
              {coldSolutionSteps.map((step, i) => (
                <li key={i} dangerouslySetInnerHTML={{ __html: renderMath(step) }} />
              ))}
            </ol>
          </details>
        )}
        <button
          className="btn-primary ss-intro-btn"
          onClick={() => { if (sessionId !== null) void loadNext(sessionId) }}
        >
          Start practicing → <span className="ss-kbd-hint">Enter</span>
        </button>
      </div>
    )
  }

  if (phase === 'done') {
    return (
      <div className="ss-done">
        <h2>All caught up!</h2>
        <p>No concepts due right now. Check back later.</p>
        <button className="btn-primary" onClick={handleNext}>Check again</button>
      </div>
    )
  }

  if (!problem) return null

  const correct = feedback?.correct_answer ?? null

  const choiceClass = (c: string) => {
    if (phase !== 'answered') return selected === c ? 'btn-choice selected' : 'btn-choice'
    if (c === correct) return 'btn-choice correct'
    if (c === selected && feedback?.is_correct === false) return 'btn-choice wrong'
    return 'btn-choice'
  }

  const timerStr = `${Math.floor(elapsedS / 60)}:${String(elapsedS % 60).padStart(2, '0')}`
  const timerCls = elapsedS > 360 ? 'ss-timer danger' : elapsedS > 180 ? 'ss-timer warning' : 'ss-timer'
  const timeTakenS = Math.floor((Date.now() - shownAt.current) / 1000)
  const timeTakenStr = `${Math.floor(timeTakenS / 60)}:${String(timeTakenS % 60).padStart(2, '0')}`
  const timeTakenCls = timeTakenS > 360 ? 'ss-time-taken danger' : timeTakenS > 180 ? 'ss-time-taken warning' : 'ss-time-taken'

  return (
    <div className="ss-wrap">
      <div className="ss-meta">
        <span className="ss-concept">{problem.concept_name}</span>
        {phase === 'question' && (
          <span className={timerCls}>{timerStr}</span>
        )}
      </div>

      {/* Theory accordion — open first time, collapsed on repeat visits */}
      {theoryMd && (
        <details
          className="ss-theory"
          open={!localStorage.getItem(`theory-seen:${problem.concept_id}`)}
          onToggle={(e) => {
            if ((e.currentTarget as HTMLDetailsElement).open) return
            if (problem.concept_id)
              localStorage.setItem(`theory-seen:${problem.concept_id}`, '1')
          }}
        >
          <summary className="ss-theory-summary">
            📖 Learn: {problem.concept_name}
          </summary>
          <div
            className="ss-theory-body"
            dangerouslySetInnerHTML={{ __html: renderTheory(theoryMd) }}
          />
        </details>
      )}

      <div
        className="ss-statement"
        dangerouslySetInnerHTML={{ __html: renderMath(problem.statement ?? '') }}
      />

      {problem.choices ? (
        <div className="ss-choices">
          {problem.choices.map((c) => (
            <button
              key={c}
              className={choiceClass(c)}
              onClick={() => handleChoice(c)}
              disabled={phase === 'answered'}
            >
              {c}
            </button>
          ))}
        </div>
      ) : (
        <p className="ss-hint">No choices — review concept, then continue.</p>
      )}

      {phase === 'question' && hintSteps.length > 0 && hintsShown > 0 && (
        <div className="ss-hints">
          <div className="ss-hint-label">Hint ({hintsShown}/{hintSteps.length})</div>
          <ol className="ss-hint-steps">
            {hintSteps.slice(0, hintsShown).map((step, i) => (
              <li key={i} dangerouslySetInnerHTML={{ __html: renderMath(step) }} />
            ))}
          </ol>
        </div>
      )}

      {phase === 'question' && problem.choices && (
        <div className="ss-submit-row">
          <button
            className="btn-ghost ss-hint-btn"
            onClick={() => { void handleHint() }}
            disabled={hintSteps.length > 0 && hintsShown >= hintSteps.length}
          >
            {hintSteps.length === 0 ? 'Hint' : hintsShown >= hintSteps.length ? 'No more hints' : `Hint (${hintsShown}/${hintSteps.length})`}
          </button>
          <button
            className="btn-primary ss-submit"
            onClick={handleSubmit}
            disabled={selected === null}
          >
            Submit
          </button>
          <span className="ss-kbd-hint">A B C D to pick · Enter to submit / advance</span>
        </div>
      )}

      {phase === 'answered' && feedback && (
        <div className="ss-feedback">
          <div className={`ss-verdict ${feedback.is_correct ? 'correct' : feedback.is_correct === false ? 'wrong' : 'neutral'}`}>
            {feedback.is_correct === true && '✓ Correct'}
            {feedback.is_correct === false && `✗ Wrong — answer: ${correct}`}
            {feedback.is_correct === null && `Reviewed — answer: ${correct}`}
          </div>
          <div className={timeTakenCls}>
            ⏱ {timeTakenStr}{timeTakenS > 360 ? ' — over 6 min exam target' : timeTakenS > 180 ? ' — approaching limit' : ''}
          </div>

          {feedback.note && (
            <div className="ss-note">
              <strong>Note:</strong> {feedback.note}
            </div>
          )}

          {feedback.solution_steps.length > 0 && (
            <details className="ss-solution" open={feedback.is_correct === false}>
              <summary>Worked solution ({feedback.solution_steps.length} steps)</summary>
              <div
                className="ss-solution-question"
                dangerouslySetInnerHTML={{ __html: renderMath(problem.statement ?? '') }}
              />
              <ol className="ss-steps">
                {feedback.solution_steps.map((step, i) => (
                  <li
                    key={i}
                    dangerouslySetInnerHTML={{ __html: renderMath(step) }}
                  />
                ))}
              </ol>
            </details>
          )}

          <button className="btn-primary ss-next" onClick={handleNext}>
            Next →
          </button>
        </div>
      )}

      {phase === 'answered' && !problem.choices && (
        <button className="btn-primary ss-next" onClick={handleNext}>
          Next →
        </button>
      )}
    </div>
  )
}
