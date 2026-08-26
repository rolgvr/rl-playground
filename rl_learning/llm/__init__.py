"""Stage 4 -- applying RL ideas to improve a small LLM.

A large "teacher" model (via API) is distilled into a small student model
(Qwen2.5-1.5B) specialised on one task: answering questions about reinforcement
learning -- best practices, implementation steps, and which algorithms suit which
tasks/hardware. Two stages, the standard modern recipe:

    SFT  -- supervised distillation: the student copies the teacher's answers.
    DPO  -- the RL step: the teacher *judges* pairs of student answers, and the
            student is optimised to prefer the ones the judge likes.

The retrained model is then testable in a chat box (base vs tuned, side by side).
"""

from .teacher import Teacher

__all__ = ["Teacher"]
