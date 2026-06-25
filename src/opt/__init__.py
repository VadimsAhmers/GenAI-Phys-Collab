"""Multipole-channel inverse design on top of the generic `llm_opt` optimizer."""
from src.opt.multipole import (
    channel_powers,
    multipole_loss,
    feedback_string,
    validate_target,
    NAMES_E,
    NAMES_M,
    CHANNEL_ORDER,
)
from src.opt.solvers import Solver, FakeSolver, ComsolSolver
from src.opt.optimize import RadiiParametrization, MultipoleChannelObjective
from src.opt.reporter import MultipoleReporter

__all__ = [
    "channel_powers",
    "multipole_loss",
    "feedback_string",
    "validate_target",
    "NAMES_E",
    "NAMES_M",
    "CHANNEL_ORDER",
    "Solver",
    "FakeSolver",
    "ComsolSolver",
    "RadiiParametrization",
    "MultipoleChannelObjective",
    "MultipoleReporter",
]
