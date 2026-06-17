import { useEffect, useState } from 'react'
import { api } from '../api'
import type { CardState, DKTStatus, JournalStats, ReadinessOut, TimingEntry, TrainOut } from '../types'
import './Dashboard.css'

function stateTag(state: string) {
  const cls =
    state === 'new' ? 'tag tag-new'
    : state === 'learning' || state === 'relearning' ? 'tag tag-learning'
    : 'tag tag-review'
  return <span className={cls}>{state}</span>
}

/** Bar capped at 30 days — FSRS stability beyond 30d gets no visual distinction. */
function stabilityBar(stability: number | null) {
  if (stability === null) return <span className="db-bar-empty">—</span>
  const pct = Math.min(100, (stability / 30) * 100)
  return (
    <div className="db-bar-wrap" title={`${stability.toFixed(1)} days`}>
      <div className="db-bar" style={{ width: `${pct}%` }} />
      <span className="db-bar-label">{stability.toFixed(1)}d</span>
    </div>
  )
}

function formatDue(due: string | null) {
  if (!due) return <span className="muted">—</span>
  const d = new Date(due)
  const now = new Date()
  const diffH = (d.getTime() - now.getTime()) / 3_600_000
  if (diffH < 0) return <span className="overdue">overdue</span>
  if (diffH < 24) return <span className="due-soon">{diffH.toFixed(1)}h</span>
  return <span className="muted">{Math.ceil(diffH / 24)}d</span>
}

/** SVG half-arc gauge. 157 ≈ π × 50, the circumference of the r = 50 arc path. */
function ReadinessGauge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = score >= 0.70 ? 'var(--green)' : score >= 0.50 ? 'var(--yellow)' : 'var(--red)'
  return (
    <div className="db-gauge">
      <svg viewBox="0 0 120 70" width="160">
        <path d="M10,60 A50,50 0 0,1 110,60" fill="none" stroke="var(--surface2)" strokeWidth="10" strokeLinecap="round"/>
        <path
          d="M10,60 A50,50 0 0,1 110,60"
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${157 * score} 157`}
        />
        <text x="60" y="58" textAnchor="middle" fill={color} fontSize="18" fontWeight="700">{pct}%</text>
        <text x="60" y="68" textAnchor="middle" fill="var(--muted)" fontSize="7">readiness</text>
      </svg>
    </div>
  )
}

export default function Dashboard() {
  const [states, setStates] = useState<CardState[]>([])
  const [dkt, setDkt] = useState<DKTStatus | null>(null)
  const [readiness, setReadiness] = useState<ReadinessOut | null>(null)
  const [journal, setJournal] = useState<JournalStats | null>(null)
  const [timing, setTiming] = useState<TimingEntry[]>([])
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState<TrainOut | null>(null)
  const [loading, setLoading] = useState(true)
  const [sortKey, setSortKey] = useState<'state' | 'stability' | 'due' | 'reps'>('due')
  const [examScore, setExamScore] = useState('')
  const [examNotes, setExamNotes] = useState('')
  const [examSubmitting, setExamSubmitting] = useState(false)
  const [chartTs, setChartTs] = useState(Date.now())

  const refresh = async () => {
    setLoading(true)
    try {
      const [{ states: statesResult }, dktStatus, readinessResult, journalResult, timingResult] = await Promise.all([
        api.getState(),
        api.getDKTStatus(),
        api.getReadiness(false),
        api.getJournal(),
        api.getTiming(),
      ])
      setStates(statesResult)
      setDkt(dktStatus)
      setReadiness(readinessResult)
      setJournal(journalResult.stats)
      setTiming(timingResult)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])

  const handleTrain = async () => {
    setTraining(true)
    setTrainResult(null)
    try {
      const result = await api.train()
      setTrainResult(result)
      await refresh()
    } finally {
      setTraining(false)
    }
  }

  const handleSampleExam = async () => {
    const score = parseFloat(examScore)
    if (isNaN(score) || score < 0 || score > 1) return
    setExamSubmitting(true)
    try {
      await api.postSampleExam({ score, notes: examNotes || undefined })
      setExamScore('')
      setExamNotes('')
      setChartTs(Date.now())
      await refresh()
    } finally {
      setExamSubmitting(false)
    }
  }

  const sorted = [...states].sort((a, b) => {
    if (sortKey === 'stability') return (b.stability ?? -1) - (a.stability ?? -1)
    if (sortKey === 'reps') return b.reps - a.reps
    if (sortKey === 'due') {
      if (!a.due && !b.due) return 0
      if (!a.due) return 1
      if (!b.due) return -1
      return new Date(a.due).getTime() - new Date(b.due).getTime()
    }
    const order = { relearning: 0, learning: 1, new: 2, review: 3 }
    return (order[a.state as keyof typeof order] ?? 9) - (order[b.state as keyof typeof order] ?? 9)
  })

  const totalAnswered = dkt?.n_interactions ?? 0
  const gateTarget = dkt?.dkt_min_interactions ?? 300
  const gateProgress = Math.min(100, (totalAnswered / gateTarget) * 100)

  return (
    <div className="db-wrap">

      {/* Top row: gauge + journal stats */}
      <div className="db-top-row">
        <section className="db-card db-gauge-card">
          <h2 className="db-heading">Readiness</h2>
          {readiness && <ReadinessGauge score={readiness.score} />}
          <div className="db-readiness-detail">
            {readiness && Object.entries(readiness.detail).map(([cat, categoryDetail]) => (
              <div key={cat} className="db-cat-row">
                <span className="db-cat-name">{cat}</span>
                <div className="db-cat-bar-wrap">
                  <div className="db-cat-bar" style={{ width: `${categoryDetail.score * 100}%`,
                    background: categoryDetail.score >= 0.70 ? 'var(--green)' : categoryDetail.score >= 0.50 ? 'var(--yellow)' : 'var(--red)'
                  }} />
                </div>
                <span className="db-cat-pct">{Math.round(categoryDetail.score * 100)}%</span>
              </div>
            ))}
          </div>
        </section>

        <section className="db-card db-stats-card">
          <h2 className="db-heading">Study Stats</h2>
          {loading || !journal ? <div className="muted">Loading…</div> : (
            <div className="db-stats">
              <div className="db-stat"><span className="db-stat-val">{journal.total_answered}</span><span className="db-stat-label">answered</span></div>
              <div className="db-stat"><span className="db-stat-val">{Math.round(journal.accuracy * 100)}%</span><span className="db-stat-label">accuracy</span></div>
              <div className="db-stat"><span className="db-stat-val">{journal.total_hours}h</span><span className="db-stat-label">studied</span></div>
              <div className="db-stat"><span className="db-stat-val">{journal.sessions_completed}</span><span className="db-stat-label">sessions</span></div>
              <div className="db-stat"><span className="db-stat-val">{journal.concepts_touched}</span><span className="db-stat-label">concepts touched</span></div>
            </div>
          )}
        </section>
      </div>

      {/* Charts */}
      <section className="db-section">
        <h2 className="db-heading">Readiness Over Time</h2>
        <img
          className="db-chart"
          src={`${api.chartUrl('timeline')}?t=${chartTs}`}
          alt="Readiness over time"
          onLoad={() => {}}
        />
      </section>

      <section className="db-section">
        <h2 className="db-heading">Per-Category Mastery</h2>
        <img
          className="db-chart"
          src={`${api.chartUrl('category')}?t=${chartTs}`}
          alt="Category mastery"
        />
      </section>

      {/* Sample exam entry */}
      <section className="db-card">
        <h2 className="db-heading">Log Sample Exam</h2>
        <div className="db-exam-form">
          <input
            className="db-input"
            type="number"
            min="0" max="1" step="0.01"
            placeholder="Score (0–1, e.g. 0.72)"
            value={examScore}
            onChange={(e) => setExamScore(e.target.value)}
          />
          <input
            className="db-input"
            type="text"
            placeholder="Notes (optional)"
            value={examNotes}
            onChange={(e) => setExamNotes(e.target.value)}
          />
          <button
            className="btn-primary"
            onClick={handleSampleExam}
            disabled={examSubmitting || !examScore}
          >
            {examSubmitting ? 'Saving…' : 'Log exam'}
          </button>
        </div>
        <div className="db-section" style={{ marginTop: 12 }}>
          <img
            className="db-chart"
            src={`${api.chartUrl('predicted-vs-actual')}?t=${chartTs}`}
            alt="Predicted vs actual"
          />
        </div>
      </section>

      {/* DKT Status */}
      <section className="db-card">
        <h2 className="db-heading">DKT Status</h2>
        {loading ? <div className="muted">Loading…</div> : dkt && (
          <div className="db-dkt">
            <div className="db-dkt-row">
              <span className="db-dkt-label">Active</span>
              <span className={dkt.dkt_active ? 'chip-green' : 'chip-muted'}>
                {dkt.dkt_active ? 'yes' : 'no'}
              </span>
            </div>
            <div className="db-dkt-row">
              <span className="db-dkt-label">Interactions</span>
              <span>{totalAnswered} / {gateTarget}</span>
            </div>
            <div className="db-gate-bar-wrap">
              <div className="db-gate-bar" style={{ width: `${gateProgress}%` }} />
            </div>
            {dkt.checkpoint && (
              <div className="db-dkt-row">
                <span className="db-dkt-label">Best val-AUC</span>
                <span>{dkt.checkpoint.val_auc.toFixed(3)} (epoch {dkt.checkpoint.epoch})</span>
              </div>
            )}
            <div className="db-train-row">
              <button className="btn-ghost" onClick={handleTrain} disabled={training}>
                {training ? 'Training…' : 'Train DKT'}
              </button>
              {trainResult && (
                <span className="db-train-result">
                  {trainResult.error
                    ? <span className="overdue">{trainResult.error}</span>
                    : <>AUC {trainResult.val_auc?.toFixed(3)} · epoch {trainResult.best_epoch}</>
                  }
                </span>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Timing Analytics */}
      {timing.length > 0 && (
        <section className="db-section">
          <h2 className="db-heading">Speed per Concept <span className="db-heading-sub">target: under 6 min</span></h2>
          <div className="db-table-wrap">
            <table className="db-table">
              <thead>
                <tr>
                  <th>Concept</th>
                  <th>Avg time</th>
                  <th>Answered</th>
                  <th>% over target</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {timing.map((entry) => {
                  const cls = entry.avg_s > 360 ? 'timing-danger'
                    : entry.avg_s > 180 ? 'timing-warning'
                    : 'timing-ok'
                  const mins = Math.floor(entry.avg_s / 60)
                  const secs = String(Math.round(entry.avg_s % 60)).padStart(2, '0')
                  return (
                    <tr key={entry.concept_id}>
                      <td className="db-concept-name">{entry.concept_name}</td>
                      <td className={`mono ${cls}`}>{mins}:{secs}</td>
                      <td className="mono">{entry.n_answered}</td>
                      <td className={`mono ${entry.pct_over_target > 0 ? cls : ''}`}>
                        {entry.pct_over_target > 0 ? `${entry.pct_over_target}%` : '—'}
                      </td>
                      <td>
                        <div className="db-timing-bar-wrap">
                          <div
                            className={`db-timing-bar ${cls}`}
                            style={{ width: `${Math.min(100, (entry.avg_s / 360) * 100)}%` }}
                          />
                          <div className="db-timing-target" />
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Concept States */}
      <section className="db-section">
        <div className="db-section-header">
          <h2 className="db-heading">Concepts ({states.length})</h2>
          <select
            className="db-sort"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as typeof sortKey)}
          >
            <option value="due">Sort: due</option>
            <option value="state">Sort: state</option>
            <option value="stability">Sort: stability</option>
            <option value="reps">Sort: reps</option>
          </select>
        </div>
        {loading ? (
          <div className="muted">Loading…</div>
        ) : (
          <div className="db-table-wrap">
            <table className="db-table">
              <thead>
                <tr>
                  <th>Concept</th>
                  <th>State</th>
                  <th>Reps</th>
                  <th>Lapses</th>
                  <th>Stability</th>
                  <th>Due in</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((cardState) => (
                  <tr key={cardState.concept_id}>
                    <td className="db-concept-name">{cardState.concept_name}</td>
                    <td>{stateTag(cardState.state)}</td>
                    <td className="mono">{cardState.reps}</td>
                    <td className="mono">{cardState.lapses > 0 ? <span className="overdue">{cardState.lapses}</span> : 0}</td>
                    <td>{stabilityBar(cardState.stability)}</td>
                    <td>{formatDue(cardState.due)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
