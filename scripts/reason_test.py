"""Verify the reasoning-graph executor runs a generate->critique->refine loop."""
from rl_learning import reasoning

graph = {
    "nodes": [
        {"id": "q", "type": "input"},
        {"id": "g", "type": "generator", "model": "local"},
        {"id": "c", "type": "relevance_critic", "model": "local", "threshold": 3},
        {"id": "r", "type": "refiner", "model": "local"},
        {"id": "o", "type": "output"},
    ],
    "links": [
        {"from": "q", "port": "out", "to": "g"},
        {"from": "g", "port": "out", "to": "c"},
        {"from": "c", "port": "pass", "to": "o"},
        {"from": "c", "port": "fail", "to": "r"},
        {"from": "r", "port": "out", "to": "c"},
    ],
}

res = reasoning.run_graph(graph, "Why does DQN use a target network?", max_steps=10)
print("iterations (refine loops):", res["iterations"])
for i, s in enumerate(res["steps"]):
    extra = ""
    if s["type"].endswith("critic"):
        extra = f" [score {s['score']}/{s['threshold']} -> {'PASS' if s['passed'] else 'FAIL'}]"
    print(f"{i+1}. {s['label']}{extra}: {s.get('text','')[:90]}")
print("\nFINAL:", res["final"][:160])
