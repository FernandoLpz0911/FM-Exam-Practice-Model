"""Generation package — importing sub-modules runs @register decorators, populating _generators."""
from engine.generation import annuity, bond, derivatives, duration, interest, loan  # noqa: F401
from engine.generation.base import generate, list_kinds

__all__ = ["generate", "list_kinds"]
