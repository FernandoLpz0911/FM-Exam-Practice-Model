export interface NextOut {
  item_id: number | null
  concept_id: string | null
  concept_name: string | null
  statement: string | null
  choices: string[] | null
  message: string | null
  is_new_concept: boolean | null
}

export interface AnswerOut {
  is_correct: boolean | null
  correct_answer: string | null
  note: string | null
  solution_steps: string[]
  next_review_at: string
}

export interface CardState {
  concept_id: string
  concept_name: string
  reps: number
  lapses: number
  stability: number | null
  due: string | null
  state: string
}

export interface DKTStatus {
  n_interactions: number
  dkt_min_interactions: number
  dkt_min_auc: number
  checkpoint: {
    epoch: number
    val_auc: number
    n_concepts: number
  } | null
  dkt_active: boolean
}

export interface ReadinessOut {
  score: number
  dkt_active: boolean
  detail: Record<string, {
    score: number
    band_weight: number
    concepts: { id: string; name: string; p_correct: number }[]
  }>
}

export interface JournalStats {
  total_answered: number
  total_correct: number
  accuracy: number
  total_hours: number
  sessions_completed: number
  concepts_touched: number
}

export interface SampleExamIn {
  score: number
  n_questions?: number
  passing_score?: number
  notes?: string
}

export interface MockExamProblem {
  item_id: number
  concept_id: string
  concept_name: string
  category: string
  statement: string
  choices: string[]
}

export interface MockExamResult {
  score: number
  n_correct: number
  n_total: number
  elapsed_s: number
  by_category: Record<string, { correct: number; total: number }>
  results: {
    item_id: number
    correct_answer: string | null
    user_answer: string
    is_correct: boolean
    concept_id: string | null
  }[]
}

export interface ConceptDetail {
  id: string
  name: string
  category: string
  exam_weight_tier: number
  summary: string | null
  prerequisites: string[]
  theory_md: string | null
}

export interface TimingEntry {
  concept_id: string
  concept_name: string
  avg_s: number
  n_answered: number
  pct_over_target: number
}

export interface TrainOut {
  val_auc: number | null
  best_epoch: number | null
  n_interactions: number
  epochs_run: number
  checkpoint_path: string | null
  dkt_active: boolean
  error: string | null
}
