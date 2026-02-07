"""NATO two-dimensional confidence rating model.

Dimension 1 (Reliability) — rates the source itself (letter A-F).
Dimension 2 (Credibility) — rates a specific claim from that source (number 1-6).

Combined rating is ``<letter><number>``, e.g. ``B2``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel
from pydantic import Field


class Reliability(str, Enum):
    """NATO source reliability rating (Dimension 1).

    A = Completely reliable, F = Reliability cannot be judged.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class Credibility(str, Enum):
    """NATO information credibility rating (Dimension 2).

    1 = Confirmed, 6 = Truth cannot be judged.
    """

    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"


# Ordered lists for comparison operations
_RELIABILITY_ORDER: list[Reliability] = list(Reliability)
_CREDIBILITY_ORDER: list[Credibility] = list(Credibility)


class NATORating(BaseModel):
    """NATO two-dimensional confidence rating (e.g. B2).

    Immutable value object combining source reliability and information
    credibility into a single rating.
    """

    model_config = {"frozen": True}

    reliability: Reliability = Field(
        description="Source reliability letter (A-F).",
    )
    credibility: Credibility = Field(
        description="Information credibility number (1-6).",
    )

    def __str__(self) -> str:
        return f"{self.reliability.value}{self.credibility.value}"

    def __repr__(self) -> str:
        return f"NATORating('{self}')"

    @classmethod
    def from_str(cls, value: str) -> NATORating:
        """Parse a rating string like ``'B2'`` into a ``NATORating``."""
        if not value or len(value) < 2:
            msg = f"Invalid NATO rating: {value!r}"
            raise ValueError(msg)
        letter = value[0].upper()
        number = value[1:]
        return cls(
            reliability=Reliability(letter),
            credibility=Credibility(number),
        )

    def is_better_or_equal(self, other: NATORating) -> bool:
        """Return ``True`` if this rating is at least as good as *other*.

        A rating is better when its reliability index is lower (A < F)
        **and** its credibility index is lower (1 < 6).
        """
        self_r = _RELIABILITY_ORDER.index(self.reliability)
        other_r = _RELIABILITY_ORDER.index(other.reliability)
        self_c = _CREDIBILITY_ORDER.index(self.credibility)
        other_c = _CREDIBILITY_ORDER.index(other.credibility)
        return self_r <= other_r and self_c <= other_c


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def worst_reliability(*ratings: Reliability) -> Reliability:
    """Return the worst (highest-index) reliability letter.

    Raises ``ValueError`` if called with no arguments.
    """
    if not ratings:
        msg = "worst_reliability() requires at least one argument"
        raise ValueError(msg)
    worst_idx = max(_RELIABILITY_ORDER.index(r) for r in ratings)
    return _RELIABILITY_ORDER[worst_idx]


def degrade_credibility(credibility: Credibility, steps: int = 1) -> Credibility:
    """Shift credibility toward worse by *steps*, clamped at ``SIX``."""
    idx = _CREDIBILITY_ORDER.index(credibility)
    new_idx = min(idx + steps, len(_CREDIBILITY_ORDER) - 1)
    return _CREDIBILITY_ORDER[new_idx]


def credibility_from_int(value: int) -> Credibility:
    """Map integer 1-6 to the corresponding ``Credibility`` enum member."""
    if not 1 <= value <= 6:
        msg = f"Credibility value must be 1-6, got {value}"
        raise ValueError(msg)
    return _CREDIBILITY_ORDER[value - 1]
