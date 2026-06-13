# CLAUDE.md — FM Engine Project Context

## 1. What this project is

Adaptive exam-prep engine for SOA **Exam FM** (Financial Mathematics).
Separate project from the P engine — zero risk of breaking P engine while studying for P.
Start practicing FM only after passing P.

Architecture mirrors the P engine exactly. Same FSRS + DKT core, same React frontend.
FM-specific: different concept graph, different generators, different solved solutions.

## 2. Hard constraints (same as P engine)

- **Local-first.** No mandatory cloud.
- **No LLM in the core.** Deterministic worked solutions only.
- **GPU budget: NVIDIA GTX 1660 Super, 6 GB VRAM.**
- **Answers and solutions are computed.** Generator and solver share the same math.
- **Reproducibility.** Seed RNGs. Log every interaction.

## 3. Ports

| Service | Port |
|---|---|
| FM FastAPI engine | **8001** |
| FM React frontend | **5174** |
| P FastAPI engine | 8000 |
| P React frontend | 5173 |

Never run both engines simultaneously unless you have the VRAM/CPU budget.

## 4. Tech stack

Same as P engine: Python 3.11 + FastAPI, PyTorch, SQLite, React + TypeScript + Vite.

## 5. FM concept categories

| Category | Concepts |
|---|---|
| interest | TVM basics, nominal/effective, force of interest, discount rate |
| annuity | Immediate, due, perpetuity, deferred, varying (arithmetic/geometric) |
| loan | Amortization, interest/principal split, sinking fund |
| bond | Price formula, Makeham, premium/discount |
| duration | Macaulay, modified, convexity, Redington immunization |
| derivatives | Spot/forward rates, forward contracts, options, put-call parity, swaps |

## 6. Generator → solver → misconception chain

For every `@register("kind")` in `engine/generation/`, there is a matching
`@_reg("kind")` in `engine/feedback/solve.py` and entries in
`engine/feedback/misconceptions.py`.
**Never add a generator without adding the solver and misconception notes.**

## 7. Running the engine

```bash
cd /home/fez/Documents/ProgrammingProjects/ExamFMEngine

# First-time setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m engine.db.seed            # seed SQLite from concept_graph.seed.json

# Start engine
uvicorn engine.main:app --port 8001 --reload

# Start frontend (separate terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5174
```

## 8. Key files

- `engine/main.py` — FastAPI entrypoint, port 8001
- `engine/generation/` — FM generators (interest, annuity, loan, bond, duration, derivatives)
- `engine/feedback/solve.py` — Worked solutions mirroring generators
- `engine/feedback/misconceptions.py` — Misconception notes keyed by (kind, ask)
- `data/concept_graph.seed.json` — 24 FM concept nodes
- `data/concept_theory.json` — Theory markdown per concept
- `frontend/vite.config.ts` — Proxies /api → localhost:8001
