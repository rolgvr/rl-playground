"""The task taxonomy + prompt templates the teacher uses to build the dataset.

We keep the scope tight (this is what makes a *small* model good): question/answer
pairs about reinforcement learning practice. No live news — knowledge is baked in
as of the teacher's training, exactly as the user chose.
"""

# Sub-topics the student should become good at. Each seeds the teacher to write
# varied, realistic user questions and high-quality answers.
TOPICS = {
    "best_practices": "Best practices for training RL agents: reward shaping, exploration vs exploitation, "
                      "hyperparameter choices, debugging unstable training, evaluation, reproducibility.",
    "implementation_steps": "Step-by-step implementation guidance: how to implement DQN, PPO, A2C, SARSA, "
                            "replay buffers, target networks, GAE, the training loop, common pitfalls.",
    "algorithm_task_fit": "Which algorithm fits which task: discrete vs continuous actions, sparse vs dense "
                          "rewards, sample efficiency, on- vs off-policy, when to use value-based vs policy-gradient.",
    "algorithm_hardware_fit": "Compatibility with hardware/compute: what runs on CPU vs a single GPU vs many, "
                              "memory needs of replay buffers and large nets, when tabular suffices, scaling considerations.",
    "concepts": "Clear explanations of core concepts: value functions, advantage, the Bellman equation, "
                "policies, the exploration problem, credit assignment, discounting.",
}

SYSTEM_TEACHER = (
    "You are an expert reinforcement-learning engineer writing training data to teach a small model. "
    "Write accurate, concrete, practical answers a practitioner would value. Prefer specifics over fluff."
)


def qa_prompt(topic_key: str, n: int) -> str:
    desc = TOPICS[topic_key]
    return (
        f"Generate {n} diverse question-answer pairs about: {desc}\n\n"
        "Vary the phrasing and difficulty as a real user would ask. Answers should be 2-5 sentences, "
        "correct and actionable. Return ONLY a JSON array of objects with keys \"question\" and \"answer\". "
        "No markdown, no commentary."
    )


JUDGE_SYSTEM = (
    "You are a strict expert judge of reinforcement-learning answers. Pick the response that is more correct, "
    "concrete and helpful. Reply with ONLY the single letter A or B."
)


def judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Response A:\n{answer_a}\n\n"
        f"Response B:\n{answer_b}\n\n"
        "Which response is better? Reply with only 'A' or 'B'."
    )


SCORE_SYSTEM = (
    "You are a strict grader of reinforcement-learning answers. Rate the answer's correctness and "
    "helpfulness for a practitioner on an integer scale from 1 (wrong or useless) to 5 (excellent). "
    "Reply with ONLY the integer 1-5."
)


def score_prompt(question: str, answer: str) -> str:
    return f"Question: {question}\n\nAnswer:\n{answer}\n\nScore (1-5):"
