from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .actions import Action, legal_actions
from .constraints import joint_action_legal
from .evaluator import Score, evaluate
from .priority import build_priority
from .state import WorldState
from .transition import transition

Evaluator = Callable[[WorldState, int], Score]


@dataclass(frozen=True)
class DecisionRecord:
    agent_id: int
    candidates: tuple[tuple[str, tuple[int, int, int, int]], ...]
    chosen: str


@dataclass(frozen=True)
class CoordinationResult:
    actions: Dict[int, Action]
    decisions: tuple[DecisionRecord, ...]
    priority: tuple[int, ...]
    next_state: WorldState
    score: Score
    accepted: bool
    rejection_reason: str | None
    communication_events: tuple[str, ...]


def _movement_cost(actions: Dict[int, Action]) -> int:
    return sum(1 for action in actions.values() if action.kind in {"UP", "DOWN", "LEFT", "RIGHT"})


def coordinate_once(state: WorldState, evaluator: Evaluator = evaluate) -> CoordinationResult:
    priority = tuple(build_priority(state))
    committed: Dict[int, Action] = {}
    decisions: list[DecisionRecord] = []
    communications: list[str] = [
        f"STATE_BROADCAST agents={len(state.agents)} timestep={state.timestep}",
        f"PRIORITY_BROADCAST order={list(priority)}",
    ]

    for agent_id in priority:
        candidates = legal_actions(state, agent_id, committed=committed)
        scored: list[tuple[Action, Score]] = []

        if not candidates:
            raise RuntimeError(f"Agent {agent_id} has no legal action under current commitments")

        for action in candidates:
            hypothetical = dict(committed)
            hypothetical[agent_id] = action
            if len(hypothetical) == len(state.agents):
                legal, _ = joint_action_legal(state, hypothetical)
                if not legal:
                    continue
                candidate_state = transition(state, hypothetical)
            else:
                # For partial commitment, hold uncommitted agents stationary for evaluation.
                provisional = dict(hypothetical)
                for other in state.agents:
                    provisional.setdefault(other.agent_id, Action("WAIT"))
                legal, _ = joint_action_legal(state, provisional)
                if not legal:
                    continue
                candidate_state = transition(state, provisional)

            score = evaluator(candidate_state, state.timestep + _movement_cost(hypothetical))
            scored.append((action, score))

        if not scored:
            raise RuntimeError(f"Agent {agent_id} has no jointly legal candidate action")

        action_preference = {
            "DROP": 7,
            "PICKUP": 6,
            "UP": 5,
            "DOWN": 4,
            "LEFT": 3,
            "RIGHT": 2,
            "WAIT": 1,
        }
        scored.sort(
            key=lambda pair: (pair[1].as_tuple(), action_preference.get(pair[0].kind, 0)),
            reverse=True,
        )
        chosen, _ = scored[0]
        committed[agent_id] = chosen
        decisions.append(
            DecisionRecord(
                agent_id=agent_id,
                candidates=tuple((a.kind, s.as_tuple()) for a, s in scored),
                chosen=chosen.kind,
            )
        )
        communications.append(f"DECISION_BROADCAST agent={agent_id} action={chosen.kind}")

    next_state = transition(state, committed)
    score = evaluator(next_state, state.timestep + _movement_cost(committed))
    accepted = score.as_tuple() >= evaluator(state, state.timestep).as_tuple() or next_state.goal_reached
    reason = None if accepted else "Global score did not improve"
    communications.append(f"VERIFIER score={score.scalar():.1f} accepted={accepted}")

    return CoordinationResult(
        actions=committed,
        decisions=tuple(decisions),
        priority=priority,
        next_state=next_state,
        score=score,
        accepted=accepted,
        rejection_reason=reason,
        communication_events=tuple(communications),
    )
