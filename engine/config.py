"""Runtime configuration — all values overridable via environment variables."""
import os

TARGET_RETENTION: float = float(os.getenv("TARGET_RETENTION", "0.9"))
MASTERY_STABILITY_DAYS: float = float(os.getenv("MASTERY_STABILITY_DAYS", "21.0"))
MASTERY_MIN_REPS: int = int(os.getenv("MASTERY_MIN_REPS", "3"))
DKT_MIN_INTERACTIONS: int = int(os.getenv("DKT_MIN_INTERACTIONS", "300"))
DKT_MIN_AUC: float = float(os.getenv("DKT_MIN_AUC", "0.70"))

# Early reinforcement: cap the FSRS-scheduled interval to 1 day for the first
# N reviews of any concept. Prevents "Easy" ratings from pushing new procedural
# skills to 8+ day gaps before the skill is actually encoded.
EARLY_REINFORCEMENT_REPS: int = int(os.getenv("EARLY_REINFORCEMENT_REPS", "7"))

# Prerequisite warmth: when routing, prefer concepts whose prerequisites were
# reviewed within this many days (dependency chain stays coherent in memory).
PREREQUISITE_WARMTH_DAYS: int = int(os.getenv("PREREQUISITE_WARMTH_DAYS", "3"))

# Difficulty ladder: reps thresholds for unlocking medium (tier 2) and hard
# (tier 3) problem types within a concept. Below tier2: easy only. Between
# tier2 and tier3: easy + medium. At or above tier3: all difficulty levels.
DIFFICULTY_TIER2_REPS: int = int(os.getenv("DIFFICULTY_TIER2_REPS", "3"))
DIFFICULTY_TIER3_REPS: int = int(os.getenv("DIFFICULTY_TIER3_REPS", "6"))
