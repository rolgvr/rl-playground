"""A compact Pong environment, deep-RL ready, with the same interface as Pac-Man.

The agent drives the left paddle; a simple (deliberately imperfect) AI drives the
right one. It learns from the screen: image layers for its paddle, the opponent's
paddle, and the ball. A single snapshot can't reveal which way the ball is
travelling, which would make the problem non-Markov, so we add a 4th layer for
the ball's *previous* position — a minimal stand-in for the frame-stacking real
Atari agents use. From "now" + "previous" the network can infer velocity.

Reward: +1 for scoring, −1 for being scored on, and a small +0.1 for hitting the
ball (so it first learns the basic skill of making contact, then learns to win).
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np

# channels: agent paddle, opponent paddle, ball now, ball previous
N_CHANNELS = 4

# actions: 0 = up, 1 = down, 2 = stay
ACTIONS = [-1, 1, 0]
ACTION_NAMES = ["up", "down", "stay"]

WIN_POINTS = 3


class PongGame:
    def __init__(self, rows: int = 16, cols: int = 20, paddle: int = 4, max_steps: int = 240):
        self.rows = rows
        self.cols = cols
        self.ph = paddle
        self.n_actions = 3
        self.n_channels = N_CHANNELS
        self.max_steps = max_steps
        self.agent_col = 1
        self.opp_col = cols - 2
        self.reset()

    def reset(self):
        self.agent_y = (self.rows - self.ph) // 2
        self.opp_y = (self.rows - self.ph) // 2
        self.ball = [self.rows // 2, self.cols // 2]
        self.prev_ball = list(self.ball)
        self.vel = [random.choice([-1, 1]), random.choice([-1, 1])]
        self.agent_score = 0
        self.opp_score = 0
        self.steps = 0
        self.score = 0          # net points (agent − opponent), used for the curve
        self.done = False
        self.outcome = None
        return self.observation()

    # --- helpers -----------------------------------------------------------

    def _covers(self, top, r):
        return top <= r < top + self.ph

    def _serve(self, toward_agent: bool):
        self.ball = [self.rows // 2, self.cols // 2]
        self.prev_ball = list(self.ball)
        self.vel = [random.choice([-1, 1]), -1 if toward_agent else 1]

    # --- step --------------------------------------------------------------

    def step(self, action: int):
        if self.done:
            return self.observation(), 0.0, True, {}
        self.steps += 1
        reward = 0.0

        # agent paddle
        self.agent_y = int(np.clip(self.agent_y + ACTIONS[action], 0, self.rows - self.ph))

        # opponent AI: track the ball, but imperfectly so it can be beaten
        if random.random() < 0.8:
            center = self.opp_y + self.ph // 2
            if self.ball[0] < center:
                self.opp_y -= 1
            elif self.ball[0] > center:
                self.opp_y += 1
            self.opp_y = int(np.clip(self.opp_y, 0, self.rows - self.ph))

        # move the ball
        self.prev_ball = list(self.ball)
        self.ball[0] += self.vel[0]
        self.ball[1] += self.vel[1]

        # bounce off top / bottom
        if self.ball[0] < 0:
            self.ball[0] = 0; self.vel[0] = 1
        elif self.ball[0] >= self.rows:
            self.ball[0] = self.rows - 1; self.vel[0] = -1

        # agent's side
        if self.ball[1] <= self.agent_col:
            if self._covers(self.agent_y, self.ball[0]):
                self.ball[1] = self.agent_col + 1; self.vel[1] = 1
                reward += 0.1                                  # rewarded for contact
            else:
                self.opp_score += 1; reward -= 1.0
                self._serve(toward_agent=False)
        # opponent's side
        elif self.ball[1] >= self.opp_col:
            if self._covers(self.opp_y, self.ball[0]):
                self.ball[1] = self.opp_col - 1; self.vel[1] = -1
            else:
                self.agent_score += 1; reward += 1.0
                self._serve(toward_agent=True)

        self.score = self.agent_score - self.opp_score

        if self.agent_score >= WIN_POINTS:
            self.done = True; self.outcome = "win"
        elif self.opp_score >= WIN_POINTS:
            self.done = True; self.outcome = "lose"
        elif self.steps >= self.max_steps:
            self.done = True
            self.outcome = "win" if self.agent_score > self.opp_score else "lose" if self.opp_score > self.agent_score else "draw"

        return self.observation(), reward, self.done, {}

    # --- observation / rendering ------------------------------------------

    def observation(self) -> np.ndarray:
        obs = np.zeros((N_CHANNELS, self.rows, self.cols), dtype=np.float32)
        obs[0, self.agent_y:self.agent_y + self.ph, self.agent_col] = 1.0
        obs[1, self.opp_y:self.opp_y + self.ph, self.opp_col] = 1.0
        obs[2, self.ball[0], self.ball[1]] = 1.0
        obs[3, self.prev_ball[0], self.prev_ball[1]] = 1.0
        return obs

    def frame(self) -> dict:
        return {
            "agent_y": self.agent_y, "opp_y": self.opp_y, "ph": self.ph,
            "agent_col": self.agent_col, "opp_col": self.opp_col,
            "ball": list(self.ball), "score": [self.agent_score, self.opp_score],
            "step": self.steps, "done": self.done, "outcome": self.outcome,
        }

    def static_layout(self) -> dict:
        return {"game": "pong", "rows": self.rows, "cols": self.cols,
                "ph": self.ph, "agent_col": self.agent_col, "opp_col": self.opp_col}
