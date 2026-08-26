"""A small, curated seed dataset of RL question/answer pairs.

This ships with the app so the SFT/DPO pipeline works and is verifiable even
before an API teacher key is configured. With a key, the teacher *expands* this
into a much larger set.
"""

SEED_QA = [
    {"question": "When should I use a value-based method like DQN versus a policy-gradient method like PPO?",
     "answer": "Use value-based methods (DQN and variants) for discrete action spaces where sample efficiency matters and a replay buffer helps. Use policy-gradient methods (PPO, A2C) for continuous actions, stochastic policies, or when you need stable on-policy updates. PPO is the common default for continuous control; DQN for discrete, pixel-based games."},
    {"question": "What's the difference between on-policy and off-policy learning?",
     "answer": "Off-policy methods (like Q-learning/DQN) learn about the greedy policy while behaving with an exploratory one, so they can reuse old data from a replay buffer. On-policy methods (like SARSA, PPO) learn about the policy they are currently following, so they must train on freshly collected data and tend to be more stable but less sample-efficient."},
    {"question": "How do I pick the discount factor gamma?",
     "answer": "Gamma trades off immediate versus future reward. Use 0.99 for most tasks; lower it toward 0.9 for short-horizon problems or when rewards are dense and you want faster, more myopic learning. Values too close to 1 on long episodes can make value estimates high-variance and slow to converge."},
    {"question": "My DQN isn't learning. What should I check first?",
     "answer": "Check that rewards are scaled sensibly, epsilon actually decays so the agent exploits what it learns, the target network is being synced (not too often, not never), and the replay buffer is large enough to break correlation. Also verify the loss isn't exploding (use Huber/smooth-L1 loss) and that the network sees a Markov state."},
    {"question": "Why use a target network in DQN?",
     "answer": "Bootstrapping toward a value computed by the same network you're updating creates a moving target and instability. A target network is a periodically-frozen copy that supplies the bootstrap value r + gamma*max Q_target(s'), so the learning target changes slowly and training is far more stable."},
    {"question": "What is experience replay and why does it help?",
     "answer": "Experience replay stores past transitions in a buffer and samples random minibatches to train on. It breaks the temporal correlation between consecutive steps (which would otherwise bias gradient updates) and lets each transition be reused many times, greatly improving sample efficiency for off-policy methods."},
    {"question": "What does Double DQN fix?",
     "answer": "Plain DQN's max operator systematically overestimates action values because noise in the estimates is always rounded up. Double DQN decouples selection from evaluation: the online network picks the best next action, the target network values it — r + gamma*Q_target(s', argmax Q_online). This reduces the upward bias and usually improves results."},
    {"question": "What problem does the dueling DQN architecture solve?",
     "answer": "Dueling DQN splits the network into a state-value stream V(s) and an advantage stream A(s,a), recombined as Q = V + A - mean(A). It lets the agent learn how good a state is without having to learn that separately for every action, which helps in states where the action choice barely matters."},
    {"question": "What is PPO's clipping and why does it matter?",
     "answer": "PPO maximizes a surrogate objective but clips the probability ratio between the new and old policy to [1-eps, 1+eps] (eps ~0.2). This prevents any single update from moving the policy too far, giving the data-efficiency of multiple epochs over a batch without the instability of large policy jumps."},
    {"question": "What is GAE (Generalized Advantage Estimation)?",
     "answer": "GAE estimates the advantage with an exponentially-weighted sum of n-step temporal-difference errors, controlled by lambda. Lambda near 0 is low-variance but biased (one-step), near 1 is high-variance but unbiased (Monte Carlo). Around 0.95 is a good default that balances the two for PPO and A2C."},
    {"question": "How should I handle exploration in a continuous control task?",
     "answer": "Use a stochastic policy that outputs a distribution (e.g., a Gaussian) and sample from it; the policy's entropy provides exploration, often encouraged with an entropy bonus. For deterministic methods like DDPG/TD3, add action noise (Gaussian or Ornstein-Uhlenbeck). Anneal exploration as the policy improves."},
    {"question": "When is tabular Q-learning enough, and when do I need deep RL?",
     "answer": "Tabular Q-learning works when the state space is small and discrete enough to enumerate (a few thousand states), and it's exact and fast there. Once states are too many to tabulate — images, continuous variables, combinatorial state — you need function approximation (a neural network), i.e., deep RL like DQN or PPO."},
    {"question": "What hardware do I need to train a deep RL agent?",
     "answer": "A single modern GPU is plenty for most small-to-medium deep RL (DQN/PPO on small games), and often the environment/CPU stepping is the bottleneck, not the GPU. Large replay buffers need RAM; pixel-based Atari benefits from a GPU for the CNN. Tabular and small MLP agents run fine on CPU."},
    {"question": "How do I make RL training reproducible?",
     "answer": "Seed all RNGs (Python, NumPy, the framework, and the environment), log the exact config and code version, and remember that GPU nondeterminism and asynchronous environments can still cause small variations. Report results across several seeds with mean and spread rather than a single run."},
    {"question": "What is reward shaping and what's the risk?",
     "answer": "Reward shaping adds intermediate rewards to guide learning toward the goal faster. The risk is changing the optimal policy or introducing reward hacking. Potential-based shaping (F = gamma*phi(s') - phi(s)) is provably safe — it speeds learning without changing the optimal policy."},
    {"question": "Why is my policy-gradient method so unstable / high variance?",
     "answer": "Policy gradients are inherently high-variance. Reduce it with a value-function baseline (advantage instead of raw return), normalize advantages per batch, use GAE, lower the learning rate, and add an entropy bonus to avoid premature collapse. PPO's clipping and multiple epochs also help a lot over vanilla REINFORCE."},
    {"question": "What's the difference between SARSA and Q-learning?",
     "answer": "Both are temporal-difference control methods. Q-learning is off-policy: it updates toward max_a Q(s',a), the best possible next action. SARSA is on-policy: it updates toward Q(s',a') for the action it actually takes next. SARSA learns safer policies near hazards because it accounts for its own exploration."},
    {"question": "How many environment steps or episodes does deep RL typically need?",
     "answer": "It varies widely. A small gridworld can be solved in hundreds to a few thousand episodes; Atari from pixels classically needs tens of millions of frames. Sample efficiency depends on the algorithm (off-policy with replay is usually more efficient), reward density, and the size of the state space."},
    {"question": "What is the credit assignment problem in RL?",
     "answer": "Credit assignment is figuring out which earlier actions deserve credit (or blame) for a reward that arrives much later. Discounting, bootstrapping (TD methods), and advantage estimation all help propagate that signal back; sparse, delayed rewards make it harder, which is why reward shaping or curiosity can help."},
    {"question": "Should I normalize observations and rewards?",
     "answer": "Usually yes. Normalizing observations (e.g., running mean/std) keeps network inputs well-scaled and stabilizes training; reward scaling or normalization keeps value targets in a reasonable range. Many strong PPO implementations normalize both. Be careful to update statistics online and not leak across evaluation."},
]
