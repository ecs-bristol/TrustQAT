"""Learning-rate schedules (per epoch)."""

from __future__ import annotations

from collections.abc import Callable
from math import cos, pi


def cosine_decay_restarts(
    initial_learning_rate: float,
    first_decay_steps: int,
    t_mul: float = 1.0,
    m_mul: float = 1.0,
    alpha: float = 0.0,
    alpha_steps: int = 0,
) -> Callable[[int], float]:
    """Cosine decay with warm restarts, holding ``alpha`` for ``alpha_steps`` at the end of each cycle.

    Same schedule as the JEDI-linear / TrustQAT-AE training scripts, evaluated per epoch.
    """
    if first_decay_steps <= alpha_steps:
        raise ValueError('first_decay_steps must exceed alpha_steps')

    def schedule(step: int, *_args) -> float:
        n_cycle = 1
        cycle_step = step
        cycle_len = first_decay_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
            cycle_len *= t_mul
            n_cycle += 1
        cycle_t = min(cycle_step / (cycle_len - alpha_steps), 1)
        return alpha + 0.5 * (initial_learning_rate - alpha) * (1 + cos(pi * cycle_t)) * m_mul ** max(n_cycle - 1, 0)

    return schedule


def constant(learning_rate: float) -> Callable[[int], float]:
    def schedule(step: int, *_args) -> float:
        return learning_rate

    return schedule


SCHEDULES = {'cosine_restarts': cosine_decay_restarts, 'constant': constant}


def get_schedule(name: str, **kwargs) -> Callable[[int], float]:
    if name not in SCHEDULES:
        raise KeyError(f'unknown schedule {name!r}; available: {sorted(SCHEDULES)}')
    return SCHEDULES[name](**kwargs)
