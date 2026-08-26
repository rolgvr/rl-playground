"""A fixed, held-out evaluation set — kept SEPARATE from the training/seed data.

Using questions the model never trained on is what makes the scores meaningful
(no train/test contamination), and keeping the set fixed makes scores comparable
across runs. Reference answers are used for the intrinsic metrics (perplexity,
ROUGE-L); the judge metrics don't need them.
"""

EVAL_SET = [
    {"question": "I have a robot arm with continuous torque control. Which RL algorithm family should I start with and why?",
     "reference": "Use a policy-gradient / actor-critic method such as PPO or SAC, because the action space is continuous. Value-based methods like DQN need discrete actions. PPO is a stable, common default; SAC is more sample-efficient and off-policy if you can use a replay buffer."},
    {"question": "My agent's reward goes up then suddenly collapses mid-training. What are the likely causes?",
     "reference": "Common causes are too-high a learning rate, a policy that became overconfident (entropy collapsed), an unclipped/poorly-scaled value loss, or stale data in on-policy methods. Lower the learning rate, add or increase an entropy bonus, clip gradients, and check reward scaling; for PPO ensure the clip range and number of epochs aren't too aggressive."},
    {"question": "When is Double DQN preferable to plain DQN?",
     "reference": "Use Double DQN whenever you use DQN — it decouples action selection (online net) from evaluation (target net) to reduce the overestimation bias caused by the max operator. It costs nothing extra and usually gives more accurate values and better policies, especially when Q-values are noisy."},
    {"question": "What's a quick way to sanity-check that my environment and reward are correct before training?",
     "reference": "Run a random policy and a hard-coded/scripted policy and inspect episode returns, lengths, and termination reasons. Verify rewards have the sign and scale you expect, that episodes end when they should, and that the observation is Markov. Plot a few episodes; many 'RL bugs' are actually environment or reward bugs."},
    {"question": "How do I choose the replay buffer size for DQN?",
     "reference": "Large enough to decorrelate samples and retain useful experience (often 10^5–10^6 transitions), but not so large it holds very stale, off-distribution data. If memory is tight, smaller buffers work for small problems; monitor whether old experience is hurting and tune accordingly."},
    {"question": "Is PPO on-policy or off-policy, and what does that imply for data reuse?",
     "reference": "PPO is on-policy: it must train on data collected by the current policy, so it cannot reuse a large replay buffer like off-policy methods. PPO reuses each batch for a few epochs with a clipped objective, but old data quickly becomes invalid as the policy changes, making it less sample-efficient than off-policy methods."},
    {"question": "What does the entropy bonus do in policy-gradient methods?",
     "reference": "It rewards keeping the action distribution spread out, encouraging exploration and preventing the policy from collapsing prematurely onto a single action. It's added to the loss with a small coefficient that is often annealed down over training as the policy should become more decisive."},
    {"question": "Can I run deep RL without a GPU?",
     "reference": "Yes for small problems — tabular methods and small MLP policies run fine on CPU, and often the environment stepping is the bottleneck, not the network. A GPU mainly helps with large networks or pixel inputs (CNNs). For Atari-from-pixels a GPU is strongly recommended."},
    {"question": "How should I evaluate a trained RL policy fairly?",
     "reference": "Evaluate with exploration turned off (greedy/deterministic actions or low temperature), average returns over many episodes and several random seeds, and report mean with spread. Use held-out conditions if the environment varies, and don't tune hyperparameters on the evaluation episodes."},
    {"question": "What is the bias-variance trade-off controlled by lambda in GAE?",
     "reference": "GAE blends n-step TD errors with weight lambda. Lambda near 0 uses mostly one-step bootstrapping — low variance but biased by the value estimate. Lambda near 1 approaches Monte-Carlo returns — unbiased but high variance. Around 0.95 balances the two and is a common default."},
    {"question": "My discrete-action agent has thousands of possible actions. What should I consider?",
     "reference": "Very large discrete action spaces strain value-based methods (the max over actions is expensive and estimates are noisy). Consider action embeddings, factorizing the action, masking invalid actions, or a policy-gradient approach that outputs a structured distribution. Reducing or hierarchically organizing the action space also helps."},
    {"question": "What's the difference between the value function V(s) and the Q-function Q(s,a)?",
     "reference": "V(s) is the expected return from state s under the policy; Q(s,a) is the expected return from taking action a in state s and then following the policy. They relate by V(s) = E_a[Q(s,a)] under the policy, and the advantage A(s,a) = Q(s,a) - V(s) measures how much better an action is than average."},
]
