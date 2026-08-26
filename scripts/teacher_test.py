"""Confirm the API teacher works: generate a couple of QA pairs + a judgment."""
from rl_learning.llm.teacher import Teacher, available_provider

print("provider:", available_provider())
t = Teacher()
print("using:", t.provider, t.model)

qa = t.generate_qa("best_practices", 2)
print(f"\ngenerated {len(qa)} QA pairs:")
for d in qa:
    print(" Q:", d["question"])
    print(" A:", d["answer"][:120], "...")

verdict = t.judge("What is the discount factor?",
                  "It controls how much future rewards are valued; closer to 1 is more far-sighted.",
                  "It is a number.")
print("\njudge prefers:", verdict, "(expected A)")
