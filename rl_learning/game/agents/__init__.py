"""Registry of deep-RL agents that learn to play the games.

Two families:
  * value-based (dqn.py)    -- DQN, Double, Dueling, Dueling-Double
  * policy-gradient (pg.py) -- REINFORCE, A2C, PPO

Each entry carries everything the platform needs:
  * train       -- train(env, params, on_checkpoint) -> result dict (incl. net+config)
  * label, family, blurb, info   -- for the picker and the explainer
  * hyperparams -- spec the UI renders as sliders
  * code        -- the objects whose real source is shown by the "see code" button
"""

from .dqn import train as _dqn_train, ConvQNet
from .pg import (train_reinforce, train_a2c, train_ppo, PolicyNet,
                 _run_episode, _gae)

# ---- hyperparameter specs (the UI turns these into sliders) ----
_DQN_HP = [
    {"name": "episodes", "label": "Episodes", "min": 50, "max": 4000, "step": 50, "default": 400,
     "help": "Games played while training. Pac-Man ~0.1s/ep, Pong ~0.4s/ep on a good GPU."},
    {"name": "lr", "label": "Learning rate", "min": 0.0001, "max": 0.01, "step": 0.0001, "default": 0.0005,
     "help": "Gradient step size. Too high is unstable, too low is slow."},
    {"name": "gamma", "label": "Discount γ", "min": 0.5, "max": 0.999, "step": 0.001, "default": 0.97,
     "help": "Weight on future vs immediate reward. Higher = more far-sighted."},
    {"name": "eps_end", "label": "Final exploration ε", "min": 0.0, "max": 0.3, "step": 0.01, "default": 0.05,
     "help": "Long-run chance of a random move."},
    {"name": "eps_decay_frac", "label": "Exploration decay (frac)", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.6,
     "help": "Over what fraction of training ε falls from 1.0 to its final value."},
    {"name": "batch", "label": "Batch size", "min": 16, "max": 256, "step": 16, "default": 64,
     "help": "Transitions sampled from replay per update."},
    {"name": "buffer", "label": "Replay buffer", "min": 2000, "max": 50000, "step": 1000, "default": 20000,
     "help": "How many past transitions are remembered to learn from."},
    {"name": "target_sync", "label": "Target sync (steps)", "min": 100, "max": 4000, "step": 100, "default": 800,
     "help": "How often the frozen target network is refreshed."},
]
_PG_HP = [
    {"name": "episodes", "label": "Episodes", "min": 50, "max": 4000, "step": 50, "default": 700,
     "help": "Games played while training. Policy-gradient methods are sample-hungry — they need more than DQN."},
    {"name": "lr", "label": "Learning rate", "min": 0.0001, "max": 0.01, "step": 0.0001, "default": 0.001,
     "help": "Gradient step size for the policy/value network."},
    {"name": "gamma", "label": "Discount γ", "min": 0.5, "max": 0.999, "step": 0.001, "default": 0.99,
     "help": "Weight on future vs immediate reward."},
    {"name": "entropy", "label": "Entropy bonus", "min": 0.0, "max": 0.1, "step": 0.005, "default": 0.01,
     "help": "Rewards keeping options open — prevents collapsing to one move too early."},
]
_PPO_HP = _PG_HP + [
    {"name": "gae_lambda", "label": "GAE λ", "min": 0.8, "max": 1.0, "step": 0.01, "default": 0.95,
     "help": "Bias/variance knob for advantage estimation."},
    {"name": "clip", "label": "PPO clip ε", "min": 0.05, "max": 0.4, "step": 0.05, "default": 0.2,
     "help": "How far the policy may move per update. The core PPO trick."},
    {"name": "ppo_epochs", "label": "Epochs per batch", "min": 1, "max": 10, "step": 1, "default": 4,
     "help": "How many optimisation passes over each episode of data."},
]


def _dqn_entry(variant, label, blurb, info):
    return {"train": (lambda env, params, on_checkpoint=None:
                      _dqn_train(env, variant=variant, params=params, on_checkpoint=on_checkpoint)),
            "label": label, "family": "value-based", "blurb": blurb, "info": info,
            "hyperparams": _DQN_HP, "code": [ConvQNet, _dqn_train]}


def _pg_entry(fn, label, blurb, info, hp, extra_code=()):
    return {"train": (lambda env, params, on_checkpoint=None: fn(env, params, on_checkpoint)),
            "label": label, "family": "policy-gradient", "blurb": blurb, "info": info,
            "hyperparams": hp, "code": [PolicyNet, _run_episode, *extra_code, fn]}


GAME_AGENTS = {
    "dqn": _dqn_entry("dqn", "DQN",
        "The original deep Q-network: a CNN learns Q(screen, action) with experience replay + a target net.",
        ("<b>DQN</b> learned Atari from pixels (DeepMind, 2015). A conv net maps the screen to a long-term value "
         "<i>Q</i> per move and trains toward <code>r + γ·max Q(s',a')</code>, stabilised by <b>experience replay</b> "
         "and a <b>target network</b>.")),
    "double": _dqn_entry("double", "Double DQN",
        "Fixes DQN's over-optimism by splitting which action is best from how good it is.",
        ("<b>Double DQN</b> uses the <b>online</b> net to pick the next action but the <b>target</b> net to value it: "
         "<code>r + γ·Q_target(s', argmax Q_online)</code> — curing the upward bias from the plain <code>max</code>.")),
    "dueling": _dqn_entry("dueling", "Dueling DQN",
        "Splits the net into state-value V(s) and per-action advantage A(s,a).",
        ("<b>Dueling DQN</b> computes <code>Q = V(s) + A(s,a) − mean(A)</code>, so it can judge a state's worth without "
         "learning it separately for every action — often faster.")),
    "dueling_double": _dqn_entry("dueling_double", "Dueling Double DQN",
        "Both improvements at once: the dueling architecture with the Double-DQN target.",
        ("<b>Dueling Double DQN</b> combines the value/advantage split with the decoupled target — a strong, common "
         "baseline that usually beats either trick alone.")),
    "reinforce": _pg_entry(train_reinforce, "REINFORCE",
        "The classic policy gradient: push up actions that led to high total return.",
        ("<b>REINFORCE</b> learns the policy directly. After each game it increases the probability of actions that "
         "preceded high <i>return</i> and decreases the rest, using the critic only as a baseline to cut variance. "
         "Simple, on-policy, a bit noisy."), _PG_HP),
    "a2c": _pg_entry(train_a2c, "A2C",
        "Advantage Actor-Critic: a critic bootstraps the return (GAE) to steady the policy updates.",
        ("<b>A2C</b> pairs an <b>actor</b> (the policy) with a <b>critic</b> (a value estimate). It updates toward the "
         "<i>advantage</i> — how much better an action did than the critic expected — with an entropy bonus for "
         "exploration. Lower variance than REINFORCE."), _PG_HP, extra_code=(_gae,)),
    "ppo": _pg_entry(train_ppo, "PPO",
        "Proximal Policy Optimization: the modern default — reuse data with a clipped, safe update.",
        ("<b>PPO</b> is today's go-to policy method. It reuses each batch for several epochs but <b>clips</b> how far "
         "the policy may shift (<code>ratio·A</code> vs <code>clip(ratio)·A</code>), getting the efficiency of bigger "
         "updates without the instability."), _PPO_HP, extra_code=(_gae,)),
}

__all__ = ["GAME_AGENTS"]
