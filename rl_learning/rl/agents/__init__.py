"""Registry of RL agents, mirroring the pathfinding `ALGORITHMS` registry.

Each entry's `train` function has the signature

    train(env, **params) -> RLResult

so the server can train any agent uniformly. The DQN agent is imported lazily
because it pulls in PyTorch, which we only want to load when actually used.
"""

from .q_learning import train as q_learning
from .sarsa import train as sarsa
from .expected_sarsa import train as expected_sarsa
from .value_iteration import train as value_iteration
from .policy_iteration import train as policy_iteration

# id -> (train_fn, label, family, description)
RL_AGENTS = {
    "q_learning": (q_learning, "Q-learning", "model-free TD",
        "Off-policy. Learns Q(s,a) from experience, bootstrapping off the best next action. The workhorse of tabular RL."),
    "sarsa": (sarsa, "SARSA", "model-free TD",
        "On-policy. Learns the value of the policy it actually follows, so it plays it safer under slipperiness."),
    "expected_sarsa": (expected_sarsa, "Expected SARSA", "model-free TD",
        "Like SARSA but averages over the next-action distribution — smoother, more stable, often the best-behaved learner."),
    "value_iteration": (value_iteration, "Value Iteration", "model-based",
        "Given the rules, sweeps the Bellman optimality update to the exact optimal values. The ground truth (= Dijkstra)."),
    "policy_iteration": (policy_iteration, "Policy Iteration", "model-based",
        "Given the rules, alternates evaluating a policy and improving it. Reaches the optimum in a few rounds."),
}


def load_dqn():
    """Import the PyTorch DQN agent on demand; returns its registry entry."""
    from .dqn import train as dqn
    return (dqn, "DQN", "deep RL",
            "A neural network learns Q(s,a) from the raw cell position — no table. "
            "The 'ML in the background' option; needs PyTorch.")


__all__ = ["RL_AGENTS", "load_dqn"]
