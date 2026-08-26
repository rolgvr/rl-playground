# Reinforcement Learning Playground

A step-by-step environment for learning search and reinforcement learning by
**watching** algorithms work. You design one maze in the browser, then race
several algorithms on that exact maze and see — visually — which one finds the
goal with the least effort.

## The learning ladder

The whole project is built as a ladder. Each rung reuses the rung below it: the
same `GridWorld`, the same `Trace` result format, the same web animation.

| Stage | Topic | Algorithms | Status |
|-------|-------|-----------|--------|
| **1** | Classic pathfinding | BFS, DFS, Dijkstra, A*, Greedy, Weighted A*, 3× bidirectional | ✅ built |
| **1½** | Pathfinding on real streets | all of the above, on live OpenStreetMap road graphs | ✅ built |
| **2** | Tabular RL — reward, penalty, policy | Q-learning, SARSA, Expected SARSA, Value & Policy Iteration | ✅ built |
| **3** | Deep RL — agent plays a game | **Pac-Man & Pong**: 7 agents (DQN family + REINFORCE/A2C/PPO), GPU, save/replay, view code | ✅ built |
| **4** | RL for LLMs — distillation | Distill an API teacher into **Qwen2.5-1.5B** via **SFT → DPO**; test base vs tuned + eval scorecard | ✅ built |
| **5** | Reasoning loops — agent graphs | Drag-and-drop **multi-agent reasoning** (generator→critic→refiner loops), per-agent model | ✅ built |
| **6** | University for LLMs | A **Dean** designs a curriculum, distils each subject **to mastery**, and saves them as a **per-model skill registry** (toggle on/off, retrain) | ✅ built |

### Stage 1 algorithms (all implemented)

| Family | Algorithms | Optimal? |
|--------|-----------|----------|
| Uninformed | BFS, DFS | BFS yes (steps), DFS no |
| Uniform-cost | Dijkstra | yes (cost) |
| Informed + optimal | A* | yes |
| Informed + fast | Greedy, Weighted A* (W=1.5) | no — trade quality for speed |
| Meet-in-the-middle | Bi-BFS, Bi-Dijkstra, Bi-A* | first two yes, Bi-A* near-optimal |

Deliberately deferred (more specialised; ask if you want them): **Jump Point
Search** (grid-only A* turbo), **Theta\*** (any-angle paths), **IDA\*** (memory-
lean but re-expands a lot, awkward to visualise), **D\* Lite** (dynamic
re-planning when the map changes). These add nuance but don't change the core
lessons, so they're optional add-ons.

### Stage 3 — the agent plays a game (deep RL on the GPU)

Switch to the **🎮 Play (Deep RL)** view and pick a game — **Pac-Man** or
**Pong**. The agent learns to play straight from the screen — a stack of image
layers fed to a convolutional network on your GPU. There is no table and no rules
given; it learns by playing. (Pac-Man: pellets +, power pellets ++, eaten ghost
+++, caught = game over. Pong: score +1, conceded −1, paddle-hit +0.1; beat the
AI to 3.) Pong needs to sense ball *direction*, so its observation includes the
ball's previous position — a minimal stand-in for the frame-stacking real Atari
agents use.

Pick an algorithm, tune **every hyperparameter** with the sliders, open
**"ⓘ how it works"** for the explainer, and press **Train on GPU**. You watch the
score curve climb and the agent's recorded games improve checkpoint by
checkpoint — drag the slider to see it go from dying instantly to clearing the
board.

**Seven algorithms across the two great families of deep RL:**

| Family | Algorithm | What's different |
|--------|-----------|------------------|
| Value-based | **DQN** | the original: CNN + experience replay + target net |
| Value-based | **Double DQN** | decouples action *selection* from *evaluation* (kills over-optimism) |
| Value-based | **Dueling DQN** | splits the net into "state value" + "action advantage" |
| Value-based | **Dueling Double DQN** | both improvements at once |
| Policy-gradient | **REINFORCE** | push up actions that led to high return (Monte-Carlo) |
| Policy-gradient | **A2C** | actor-critic with a bootstrapped (GAE) advantage |
| Policy-gradient | **PPO** | the modern default — reuse data with a *clipped*, safe update |

Every algorithm has a **"see the code"** button that shows its *real* source
(served live via `inspect.getsource`), and **all hyperparameters** are sliders.
After training you can **💾 save the model**; saved models persist to `models/`
and can be reloaded and watched playing any time (the **Saved models** list, with
▶ Watch / 🗑). 

This stage is deep-RL only — tabular methods can't handle these games' enormous
state. Training runs in the background; the browser polls progress. Value-based
methods are more sample-efficient here; policy-gradient methods are correct but
hungrier (their default episode count is higher). (The tabular maze-RL agents
from Stage 2 still live in `rl_learning/rl/`.)

### Stage 4 — applying RL to improve an LLM (distillation)

Switch to the **🧠 Teach LLM** view. A large **teacher** (Claude or GPT via API)
teaches a small **Qwen2.5-1.5B** to answer reinforcement-learning questions, in
the standard modern recipe:

1. **Load** the model on your GPU (LoRA fine-tuning, ~3GB VRAM).
2. **SFT** — supervised distillation: the teacher writes Q&A pairs (or use the
   built-in **seed dataset**, so this step works with *no API key*), and the
   student learns to copy them.
3. **DPO** — *the RL step*: the student samples two answers, the teacher
   **judges** which is better, and the student is optimised to prefer the
   winners (RLAIF — RL from AI feedback).
4. **Test it**: ask any RL question and see the **base vs your tuned** answer
   side by side.

Set your API key by pasting it in the UI (it picks OpenAI or Anthropic). The
teacher steps cost API tokens; SFT-on-seed-data and chat are free/local.

**Evaluating it (the hard part).** Press **📊 Intrinsic** or **⚖️ Full + judge**
to score *base vs tuned* on a **fixed held-out set** (never trained on), with the
metrics the industry actually uses:

| Metric | Needs key? | What it is |
|--------|-----------|------------|
| **Win rate** (LLM-as-judge, pairwise, position-swapped) | yes | Chatbot-Arena / AlpacaEval style |
| **Rubric 1–5** (LLM-as-judge, absolute) | yes | MT-Bench single-grading |
| **Perplexity** (held-out) | no | classic intrinsic LM metric (lower=better) |
| **ROUGE-L** vs reference | no | classic lexical overlap (higher=better) |

The judge runs at temperature 0 on a fixed set, so scores are consistent and
comparable across runs. Example result after a tiny SFT: tuned beat base on
perplexity (21.6 → 14.1) and ROUGE-L (0.118 → 0.155).

Honest caveats: a 1.5B model has a real capability ceiling and can still
hallucinate; the tiny built-in seed set (20 examples) is only enough to *shift*
its style — to genuinely improve it you need the teacher to generate a few
hundred examples (needs a valid key). Fine-tuning bakes in knowledge as of
training time — it has no "latest news" (that would need retrieval/RAG, which we
deliberately skipped).

### Stage 5 — reasoning loops (multi-agent self-refinement)

Switch to the **🔁 Reasoning Lab**. Build a graph of prompted **agents** — a
generator, a relevance critic, a refiner, a verifier, self-consistency — by
clicking them from the palette and wiring the ports. Critics route **pass ✓ /
fail ✗**; send *fail* to a refiner to form a **generate → critique → refine
loop** (the Self-Refine / Reflexion pattern). Each node has its own **prompt**
and **model** (local tuned Qwen *or* the teacher API — mix and match). Type a
question, hit **Run graph**, and watch each agent's output stream in the trace,
the critic's pass/fail verdicts, and the loop iterate until it's good enough.

This is *composing agents* (prompted LLM calls), not editing weights — the
UI-friendly way to engineer reasoning behaviour. (Improving a model's *innate*
reasoning is a deeper, training-based path — reasoning-RL / GRPO — a possible
future stage.)

## Quick start

```powershell
# one-time setup (Python 3.14 venv — transformers/torch all have 3.14 wheels)
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# PyTorch with CUDA for GPU training (CPU build also works, just slower):
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# Stage 4 (LLM distillation) extras:
.venv\Scripts\python -m pip install transformers peft anthropic openai

# run
.venv\Scripts\python -m rl_learning.server.app
```

Then open <http://127.0.0.1:5000>.

## Two ways to run it

Your GPU, your trained models and your API keys **never leave your machine** in
either mode. The only difference is who serves the web page.

### `--mode local` (default) — no cloud, no account, no AWS

The Python server serves the UI itself, so the page and the API share an origin.
Nothing to configure, no pairing, no network dependency. This is the command
above, and it is what you want for developing or for running entirely offline.

```powershell
rl-playground                      # or: python -m rl_learning.server.app
```

### `--mode worker` — the hosted UI, driving your own GPU

The UI is served from the cloud, and this process is just the engine on your
machine. The browser talks **directly** to `127.0.0.1` — training data and keys
never touch our servers, and we never pay for a GPU.

```powershell
rl-playground --mode worker --allow-origin https://rl.safezoneaitech.com
```

It prints a **pairing token**; paste that into the web app once and the browser
is bound to this machine. Two things must both hold before the worker will run
anything: the request comes from an allowed origin, *and* it carries the token.
Without both, any website you happened to visit could quietly drive your GPU and
read your keys back out.

The worker listens on loopback only. It is never exposed to your network.

## Bring your own LLM (Stages 4–6)

The "teacher" that generates data and judges answers can be a hosted API **or a
model you run yourself**. Anything that speaks the OpenAI-compatible API works —
**Ollama, LM Studio, vLLM, llama.cpp**:

```powershell
ollama pull qwen2.5:7b            # auto-detected at http://localhost:11434/v1
rl-playground
```

Point it somewhere else with `RL_LOCAL_LLM_URL`. With a local teacher, Stages 4–6
cost **nothing** and run fully offline — no key, no tokens billed. The app picks
`ANTHROPIC_API_KEY`, then `OPENAI_API_KEY`, then a detected local server; the UI
lets you override.

There are **two worlds** to race the algorithms on, switched with the toggle at
the top-left:

**▦ Grid maze** (offline, instant)
1. Paint walls (drag on the grid). Drop **Mud** to add costly terrain.
2. Move the 🟢 start and 🔴 goal wherever you like.
3. Tick the algorithms you want and press **Race selected**.
4. Watch the pale cells spread (work done) and the bright line appear (the path).

**🗺 Real streets** (live data from OpenStreetMap — needs internet)
1. Click a **start** point on the map, then a **destination** nearby.
2. Press **⬇ Fetch road network** (pulls real roads via the free Overpass API;
   can take 10–30s, occasionally longer when Overpass is busy).
3. Press **Race selected** — the same nine algorithms now search the actual
   street graph, with distances in metres. A* vs Dijkstra on real roads is the
   same lesson as on the grid, made tangible.

No API key or billing is required. Keep the two points within a neighbourhood;
far-apart points pull too much data and are rejected. If a fetch fails, Overpass
is overloaded — wait a moment and try again, or pick a slightly smaller area.

### Things to try

- **Open grid, race Dijkstra vs A\*** → identical path, but A\* expands a fraction
  of the cells. That gap *is* the value of a heuristic.
- **Add a stripe of Mud** → Dijkstra and A\* bend around it; BFS and DFS walk
  straight through, because to them every step costs the same.
- **Random maze** a few times → see how DFS often finds *a* path fast but a bad one.

## How it fits together

```
rl_learning/
  grid.py            GridWorld — the grid "world": walls, mud, start, goal
  roads.py           RoadNetwork — a real OSM street graph (Overpass API)
  trace.py           Trace — the result shape for pathfinding searches
  algorithms/        9 searches; each is solve(problem) -> Trace
  rl/                tabular maze RL (Stage 2): GridGame MDP + tabular agents
  game/
    pacman.py        PacManGame — Pac-Man as a CNN-ready (channels,H,W) screen
    pong.py          PongGame — same interface; ball-direction via prev-frame channel
    agents/
      dqn.py         DQN / Double / Dueling / Dueling-Double (conv net, GPU)
      pg.py          REINFORCE / A2C / PPO (shared actor-critic conv net)
      persistence.py save / list / load / replay trained models (models/*.pt)
  llm/                Stage 4: distill an API teacher into Qwen2.5-1.5B
    teacher.py        provider-agnostic teacher/judge (Anthropic or OpenAI)
    engine.py         load student, LoRA SFT + DPO loops, base-vs-tuned generate
    data.py / seed_data.py / topics.py   dataset gen + built-in seed Q&A
    evaluate.py / eval_set.py            held-out eval: win-rate, rubric, perplexity, ROUGE-L
  server/
    app.py           Flask: /api/solve, /api/roads, /api/rl_train, /api/game_*
    static/          the interactive browser app (vanilla JS + Leaflet)
```

Two design choices make the whole thing compose:

1. **One search interface.** Every algorithm only uses `problem.successors(node)`,
   `problem.heuristic(node)`, `.start` and `.goal`. Both `GridWorld` and
   `RoadNetwork` implement those four things, so the *same* nine algorithms run
   on a grid or on real streets with zero changes.
2. **The browser never implements an algorithm.** It only plays back a `Trace`.
   So when Stage 2 adds a Q-learning agent that produces a `Trace`, it becomes
   raceable against A* with zero frontend changes.

## Requirements

- Stage 1 runs on any Python ≥ 3.8 with Flask.
- Stages 2–3 will want Python ≥ 3.11 (for `gymnasium` / `stable-baselines3`).
