"""Verify the University: curriculum -> teach-to-mastery -> skills -> rebuild."""
from rl_learning import university

ev = []
s = university.run_university("test-grad", "reinforcement learning",
                             target=0.99, max_subjects=2, max_minutes=12, max_budget=1.0,
                             on_event=lambda e: ev.append(e["type"]), on_node=lambda d: None)
print("events:", ev)
print("skills:", [(sk["name"], sk["mastery"], sk["enabled"]) for sk in s.get("skills", [])])
print("gpa:", s.get("gpa"), "mastered:", s.get("mastered"), "/", s.get("total"))

# toggle a skill off -> rebuild from enabled
m = university.load_manifest("test-grad")
if m["skills"]:
    first = m["skills"][0]["name"]
    r = university.set_skill_enabled("test-grad", first, False)
    print(f"toggled '{first}' off -> rebuilt on {r['examples']} examples ({r['enabled']} skills enabled)")
print("manifest models:", [mm["model"] for mm in university.list_models()])
