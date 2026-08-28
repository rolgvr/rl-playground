"""Flask app: the thin bridge between the browser and the Python algorithms.

The browser owns the *interaction* (drawing the maze, animating). Python owns
the *thinking* (running each algorithm and recording a Trace). They talk through
one endpoint:

    POST /api/solve
        body: { "grid": <GridWorld.to_dict()>, "algorithms": ["bfs", "astar", ...] }
        returns: { "results": { "bfs": <Trace.to_dict()>, ... } }

Keeping all the algorithm logic server-side is deliberate: when Stage 2 adds a
Q-learning agent (which trains in Python), the frontend needs zero changes -- it
just receives another Trace to play back.

Run with:  python -m rl_learning.server.app
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, redirect, request, send_from_directory

from ..algorithms import ALGORITHMS
from ..grid import GridWorld
from ..roads import RoadNetwork, haversine
import inspect

from ..game.pacman import PacManGame
from ..game.pong import PongGame
from ..game.agents import GAME_AGENTS
from ..game.agents import persistence

from . import config, security

__version__ = "1.0.0"


def _load_env():
    """Load KEY=VALUE lines from a .env at the project root so secrets (e.g.
    OPENAI_API_KEY / ANTHROPIC_API_KEY) survive the dev-server's auto-reloads.
    Values here OVERRIDE any pre-existing (possibly stale) environment variable."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        with open(os.path.join(root, ".env"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k:
                    os.environ[k] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass


_load_env()


# Selectable games. Both expose the same env interface, so the same CNN-DQN
# agents train on either.
GAMES = {
    "pacman": (PacManGame, "Pac-Man", "Eat the pellets, dodge the ghosts."),
    "pong": (PongGame, "Pong", "Beat the AI paddle to 5 points."),
}

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# How this process is running: everything-local (default) or GPU worker paired to
# the cloud UI. See rl_learning/server/config.py.
SETTINGS = config.load()
security.init(app, SETTINGS)


@app.route("/config.js")
def config_js():
    """Tells the page which deployment it is in.

    Served by us => we are the origin => same-origin API, no pairing needed. The
    cloud deploy ships a *different* config.js (see scripts/deploy_ui.py) that
    flips RL_CLOUD_UI on, which is what makes connect.js talk to 127.0.0.1
    instead of to its own origin.
    """
    return app.response_class(
        "window.RL_CLOUD_UI = false;\n", mimetype="application/javascript",
    )


@app.route("/api/health")
def health():
    """Unauthenticated liveness probe: how the web UI discovers a local worker."""
    return jsonify({
        "ok": True,
        "service": "rl-playground-worker",
        "version": __version__,
        "mode": SETTINGS.mode,
        "requires_token": SETTINGS.is_worker,
    })


@app.route("/api/capabilities")
def capabilities():
    """What this machine can actually do, so the UI can enable/grey out stages."""
    gpu = {"available": False, "name": None, "vram_gb": None, "torch": None, "cuda": None}
    try:
        import torch

        gpu["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.update(
                available=True,
                name=props.name,
                vram_gb=round(props.total_memory / (1024 ** 3), 1),
                cuda=torch.version.cuda,
            )
    except Exception as exc:  # torch missing or a broken CUDA install
        gpu["error"] = str(exc)[:200]

    from ..llm.teacher import available_provider, discover_local_models

    return jsonify({
        "gpu": gpu,
        # Deep RL and LLM fine-tuning need real VRAM; pathfinding and tabular RL
        # do not. The UI uses this to say *why* a stage is unavailable.
        "stages": {
            "pathfinding": True,
            "tabular_rl": True,
            "deep_rl": gpu["available"],
            "llm": gpu["available"],
        },
        "local_llm": discover_local_models(),
        "teacher": available_provider(),
    })

# Fetched road networks live here, keyed by an id we hand back to the browser,
# so solving doesn't re-fetch or re-send the (potentially large) graph.
_NETWORKS = {}
_NET_SEQ = [0]

# Cap how big an area we'll fetch, so a far-apart pair of clicks can't pull a
# whole region from Overpass (slow for everyone). ~0.08 deg ~ 9 km per side.
MAX_SPAN_DEG = 0.08


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# The Learn curriculum is an Astro site (site/ -> site/dist), built by CI for the
# cloud and by `npm run build` locally. A checkout without a build (no Node, or a
# wheel install) redirects to the hosted docs, which are the same files.
DOCS_DIST = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "site", "dist")
)
DOCS_URL = "https://rl.safezoneaitech.com"


@app.route("/learn")
def learn():
    if os.path.isfile(os.path.join(DOCS_DIST, "learn.html")):
        return send_from_directory(DOCS_DIST, "learn.html")
    return redirect(DOCS_URL + "/learn", code=302)


@app.route("/learn/<path:page>")
def learn_page(page):
    # Clean URLs, same convention as CloudFront: /learn/x/y -> learn/x/y.html
    if os.path.isfile(os.path.join(DOCS_DIST, "learn", page + ".html")):
        return send_from_directory(os.path.join(DOCS_DIST, "learn"), page + ".html")
    return redirect(f"{DOCS_URL}/learn/{page}", code=302)


@app.route("/_astro/<path:asset>")
def learn_asset(asset):
    return send_from_directory(os.path.join(DOCS_DIST, "_astro"), asset)


@app.route("/play")
def play():
    return send_from_directory(STATIC_DIR, "play.html")


@app.errorhandler(404)
def not_found(_err):
    """Match the cloud deployment: unknown paths get the styled 404 page.

    API calls still get JSON, so fetch() callers never try to parse HTML.
    """
    if request.path.startswith("/api/"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(STATIC_DIR, "404.html"), 404


@app.route("/api/algorithms")
def list_algorithms():
    """Tell the UI which algorithms exist and how to describe them."""
    return jsonify({
        "algorithms": [
            {"id": key, "label": label, "family": family, "description": desc}
            for key, (_fn, label, family, desc) in ALGORITHMS.items()
        ]
    })


@app.route("/api/solve", methods=["POST"])
def solve():
    payload = request.get_json(force=True)
    grid = GridWorld.from_dict(payload["grid"])
    wanted = payload.get("algorithms") or list(ALGORITHMS.keys())

    results = {}
    for key in wanted:
        entry = ALGORITHMS.get(key)
        if entry is None:
            continue
        fn = entry[0]
        results[key] = fn(grid).to_dict()

    return jsonify({"results": results})


@app.route("/api/roads", methods=["POST"])
def fetch_roads():
    """Fetch the real road graph around two clicked points and cache it.

    Body: { "points": [[lat, lon], [lat, lon]] }
    Returns the graph for drawing (nodes + edges), the snapped start/goal node
    ids, the map bounds, and a graph_id to use when solving.
    """
    payload = request.get_json(force=True)
    (lat1, lon1), (lat2, lon2) = payload["points"]

    # Bounding box around the two points, padded so roads near the endpoints
    # aren't clipped. Reject spans that are too large to fetch politely.
    pad = 0.0025
    south, north = min(lat1, lat2) - pad, max(lat1, lat2) + pad
    west, east = min(lon1, lon2) - pad, max(lon1, lon2) + pad
    if (north - south) > MAX_SPAN_DEG or (east - west) > MAX_SPAN_DEG:
        return jsonify({"error": "Those points are too far apart — pick two spots "
                                 "within roughly the same neighbourhood."}), 400

    try:
        net = RoadNetwork.from_overpass(south, west, north, east)
    except Exception as exc:  # network/Overpass failure
        return jsonify({"error": f"Could not reach OpenStreetMap (Overpass): {exc}"}), 502

    if not net.nodes:
        return jsonify({"error": "No roads found in that area. Try a more built-up spot."}), 404

    net.start = net.nearest(lat1, lon1)
    net.goal = net.nearest(lat2, lon2)

    _NET_SEQ[0] += 1
    graph_id = f"net{_NET_SEQ[0]}"
    _NETWORKS[graph_id] = net
    while len(_NETWORKS) > 8:            # road graphs are large — keep only recent ones
        _NETWORKS.pop(next(iter(_NETWORKS)))

    return jsonify({
        "graph_id": graph_id,
        "nodes": {str(nid): [lat, lon] for nid, (lat, lon) in net.nodes.items()},
        "edges": net.edge_list(),
        "start": net.start,
        "goal": net.goal,
        "bounds": [south, west, north, east],
        "counts": {"nodes": len(net.nodes), "edges": len(net.edge_list())},
    })


@app.route("/api/solve_roads", methods=["POST"])
def solve_roads():
    """Run the selected algorithms on a previously fetched road network.

    Body: { "graph_id": "...", "algorithms": [...],
            optional "start": id, "goal": id }
    """
    payload = request.get_json(force=True)
    net = _NETWORKS.get(payload.get("graph_id"))
    if net is None:
        return jsonify({"error": "That road network expired — fetch the area again."}), 410

    if payload.get("start") is not None:
        net.start = int(payload["start"])
    if payload.get("goal") is not None:
        net.goal = int(payload["goal"])

    wanted = payload.get("algorithms") or list(ALGORITHMS.keys())
    results = {}
    for key in wanted:
        entry = ALGORITHMS.get(key)
        if entry is not None:
            results[key] = entry[0](net).to_dict()

    return jsonify({"results": results})


# (The old maze-RL endpoints /api/rl_agents and /api/rl_train were removed: that
# view was replaced by the games view. The rl_learning/rl/ module stays as the
# stage-2 teaching code.)


# ---- Pac-Man deep-RL game ----
# Training takes tens of seconds to minutes, so it runs in a background thread
# and the browser polls for progress (watching the score curve grow and the
# agent's play improve checkpoint by checkpoint).
import threading
import time

_GAME_JOBS = {}
_GAME_SEQ = [0]

# One lock guards every job-state transition, and one "GPU owner" slot makes
# training runs mutually exclusive ACROSS features (game RL, LLM jobs, University)
# — they all share the same GPU, and two concurrent trainings OOM the server.
_JOBS_LOCK = threading.Lock()
_GPU = {"owner": None}


def _gpu_claim(owner):
    """Atomically claim the GPU for a training run.
    Returns None on success, or the current owner's label if busy."""
    with _JOBS_LOCK:
        if _GPU["owner"]:
            return _GPU["owner"]
        _GPU["owner"] = owner
        return None


def _gpu_release(owner):
    with _JOBS_LOCK:
        if _GPU["owner"] == owner:
            _GPU["owner"] = None


def _clamp(v, lo, hi, default):
    try:
        v = type(default)(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


@app.route("/api/game_agents")
def game_agents():
    return jsonify({"agents": [
        {"id": k, "label": v["label"], "family": v["family"], "blurb": v["blurb"],
         "info": v["info"], "hyperparams": v["hyperparams"]}
        for k, v in GAME_AGENTS.items()
    ]})


@app.route("/api/games")
def list_games():
    return jsonify({"games": [{"id": k, "label": lbl, "blurb": blurb}
                              for k, (_cls, lbl, blurb) in GAMES.items()]})


def _game_cls(game_id):
    return GAMES.get(game_id, GAMES["pacman"])[0]


@app.route("/api/game_layout")
def game_layout():
    cls = _game_cls(request.args.get("game", "pacman"))
    return jsonify(cls().static_layout())


@app.route("/api/game_train", methods=["POST"])
def game_train():
    payload = request.get_json(force=True)
    agent_id = payload.get("agent", "double")
    game_id = payload.get("game", "pacman")
    params = payload.get("params", {})
    entry = GAME_AGENTS.get(agent_id)
    if entry is None:
        return jsonify({"error": "unknown agent"}), 400
    game_cls = _game_cls(game_id)

    # One run at a time — concurrent GPU trainings (any feature) OOM and crash.
    busy = _gpu_claim("game training")
    if busy:
        return jsonify({"error": f"The GPU is busy ({busy}) — let it finish first."}), 409

    # Free GPU memory held by previous jobs' networks (they pile up otherwise).
    for j in _GAME_JOBS.values():
        j["net"] = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    with _JOBS_LOCK:
        _GAME_SEQ[0] += 1
        job_id = f"game{_GAME_SEQ[0]}"
        job = {"status": "running", "episode": 0, "episodes": int(params.get("episodes", 600)),
               "curve": [], "checkpoints": [], "device": None, "result": None, "error": None}
        _GAME_JOBS[job_id] = job

    def run():
        try:
            env = game_cls()
            def on_ckpt(ep, episodes, curve, roll):
                job["episode"] = ep
                job["episodes"] = episodes
                job["curve"] = curve
                job["checkpoints"].append(roll)
            res = entry["train"](env, params, on_ckpt)
            job["device"] = res["device"]
            job["result"] = {k: res[k] for k in ("variant", "device", "time_ms", "episodes", "layout")}
            # keep the trained net (server-side only) so it can be saved
            job["net"] = res["net"]
            job["config"] = res["config"]
            best = res.get("best")
            job["meta"] = {"game": game_id, "agent": agent_id, "variant": res["variant"],
                           "episodes": res["episodes"], "device": res["device"],
                           "best_score": (best["score"] if best else None),
                           "created": time.strftime("%Y-%m-%d %H:%M")}
            job["status"] = "done"
        except Exception as exc:  # surface training errors to the UI
            job["status"] = "error"
            job["error"] = repr(exc)
        finally:
            _gpu_release("game training")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id, "layout": game_cls().static_layout()})


@app.route("/api/game_progress")
def game_progress():
    job = _GAME_JOBS.get(request.args.get("job_id"))
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    ck = job["checkpoints"]
    return jsonify({
        "status": job["status"], "error": job["error"],
        "episode": job["episode"], "episodes": job["episodes"], "device": job["device"],
        "curve": [c["score"] for c in job["curve"]],
        "checkpoints": [{"episode": c["episode"], "score": c["score"],
                         "outcome": c["outcome"], "steps": c["steps"]} for c in ck],
        "latest_index": len(ck) - 1,
    })


@app.route("/api/game_checkpoint")
def game_checkpoint():
    """Frames of one recorded greedy game, for the browser to animate."""
    job = _GAME_JOBS.get(request.args.get("job_id"))
    if job is None or not job["checkpoints"]:
        return jsonify({"error": "no checkpoint"}), 404
    idx = int(request.args.get("index", -1))
    c = job["checkpoints"][idx]
    return jsonify({"episode": c["episode"], "score": c["score"],
                    "outcome": c["outcome"], "frames": c["frames"]})


# ---- Stage 4: LLM distillation (SFT + DPO) ----
# One model + GPU, so one LLM operation at a time. Long ops run in a thread and
# the browser polls _LLM_JOB; chat is synchronous.
_LLM = {"provider": None, "api_key": None}
_LLM_JOB = {"status": "idle", "phase": "", "step": 0, "total": 0, "curve": [], "info": "", "error": None}
_LLM_EVAL = {"report": None}


def _llm_progress(step, total, curve_or_count, error=None):
    _LLM_JOB["step"] = step
    _LLM_JOB["total"] = total
    if isinstance(curve_or_count, list):
        _LLM_JOB["curve"] = curve_or_count[-200:]
    else:
        _LLM_JOB["info"] = f"{curve_or_count} items"
    if error:
        _LLM_JOB["error"] = error


def _start_llm_job(phase, target):
    busy = _gpu_claim(f"LLM {phase}")
    if busy:
        return False
    with _JOBS_LOCK:
        if _LLM_JOB["status"] == "running":          # belt-and-braces
            _gpu_release(f"LLM {phase}")
            return False
        _LLM_JOB.update({"status": "running", "phase": phase, "step": 0, "total": 0,
                         "curve": [], "info": "", "error": None})

    def run():
        try:
            target()
            _LLM_JOB["status"] = "done"
        except Exception as exc:
            _LLM_JOB["status"] = "error"
            _LLM_JOB["error"] = repr(exc)
        finally:
            _gpu_release(f"LLM {phase}")

    threading.Thread(target=run, daemon=True).start()
    return True


@app.route("/api/llm/status")
def llm_status():
    from ..llm import engine, data
    from ..llm.teacher import available_provider
    return jsonify({
        **engine.status(),
        "sft_examples": data.sft_count(), "pref_pairs": data.pref_count(),
        "teacher": _LLM["provider"] or available_provider(),
        "job": {k: _LLM_JOB[k] for k in ("status", "phase", "step", "total", "info", "error")},
        "curve": _LLM_JOB["curve"],
        "eval": _LLM_EVAL["report"],
    })


@app.route("/api/llm/set_key", methods=["POST"])
def llm_set_key():
    p = request.get_json(force=True)
    _LLM["provider"] = p.get("provider") or None
    _LLM["api_key"] = p.get("api_key") or None
    # quick validity probe
    try:
        from ..llm.teacher import Teacher
        Teacher(provider=_LLM["provider"], api_key=_LLM["api_key"]).chat("ping", "say ok", max_tokens=3)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)[:300]}), 400


@app.route("/api/llm/load", methods=["POST"])
def llm_load():
    from ..llm import engine
    _start_llm_job("loading model", lambda: engine.load())
    return jsonify({"ok": True})


@app.route("/api/llm/gen_sft", methods=["POST"])
def llm_gen_sft():
    from ..llm import data
    n = int(request.get_json(force=True).get("n_per_topic", 8))
    ok = _start_llm_job("generating SFT data (teacher)",
                        lambda: data.generate_sft(n, _LLM["provider"], _LLM["api_key"], _llm_progress))
    return jsonify({"ok": ok})


@app.route("/api/llm/train_sft", methods=["POST"])
def llm_train_sft():
    from ..llm import data, engine
    p = request.get_json(force=True)
    epochs, lr = int(p.get("epochs", 3)), float(p.get("lr", 1e-4))
    ok = _start_llm_job("SFT training",
                        lambda: engine.sft_train(data.load_sft(True), epochs, lr, _llm_progress))
    return jsonify({"ok": ok})


@app.route("/api/llm/gen_prefs", methods=["POST"])
def llm_gen_prefs():
    from ..llm import data
    n = int(request.get_json(force=True).get("n", 24))
    ok = _start_llm_job("judging answers (teacher) for DPO",
                        lambda: data.generate_prefs(n, _LLM["provider"], _LLM["api_key"], _llm_progress))
    return jsonify({"ok": ok})


@app.route("/api/llm/train_dpo", methods=["POST"])
def llm_train_dpo():
    from ..llm import data, engine
    p = request.get_json(force=True)
    prefs = data.load_prefs()
    if not prefs:
        return jsonify({"ok": False, "error": "no preference pairs yet — generate them first"}), 400
    ok = _start_llm_job("DPO training",
                        lambda: engine.dpo_train(prefs, int(p.get("epochs", 1)),
                                                 float(p.get("lr", 5e-5)), float(p.get("beta", 0.1)), _llm_progress))
    return jsonify({"ok": ok})


@app.route("/api/llm/evaluate", methods=["POST"])
def llm_evaluate():
    from ..llm import evaluate
    from ..llm.teacher import available_provider
    use_judge = bool(request.get_json(force=True).get("use_judge", False))
    if use_judge and not (_LLM["api_key"] or available_provider()):
        return jsonify({"ok": False, "error": "judge metrics need a valid teacher key"}), 400

    def task():
        rep = evaluate.run_eval(use_judge=use_judge, provider=_LLM["provider"],
                                api_key=_LLM["api_key"], on_progress=_llm_progress)
        _LLM_EVAL["report"] = rep

    ok = _start_llm_job("evaluating (held-out set)", task)
    return jsonify({"ok": ok})


@app.route("/api/llm/chat", methods=["POST"])
def llm_chat():
    from ..llm import engine
    q = request.get_json(force=True).get("question", "").strip()
    if not q:
        return jsonify({"error": "empty question"}), 400
    try:
        base = engine.generate(q, tuned=False)
        tuned = engine.generate(q, tuned=True) if engine.status()["has_tuned"] else None
        return jsonify({"base": base, "tuned": tuned})
    except Exception as exc:
        return jsonify({"error": repr(exc)}), 500


@app.route("/api/llm/free", methods=["POST"])
def llm_free():
    from ..llm import engine
    engine.free()
    return jsonify({"ok": True})


# ---- University: Dean designs a curriculum, students master skills ----
_UNI = {"status": "idle", "events": [], "summary": None, "stop": False, "log_path": None,
        "active": {}, "rebuild": "idle", "active_model": None, "chat": [], "thread": None,
        "control": {"guidance": [], "questions": [], "curriculum_ops": [], "overrides": {}}}


def _uni_chat(m):
    _UNI["chat"].append(m)
    if len(_UNI["chat"]) > 400:           # keep memory bounded on long runs
        del _UNI["chat"][:-400]


def _uni_alive():
    """True only if a worker thread is genuinely still running. Auto-heals a stale
    'running' status left behind by a crashed worker or an old/reloaded session."""
    t = _UNI.get("thread")
    alive = bool(t and t.is_alive())
    with _JOBS_LOCK:
        if _UNI["status"] == "running" and not alive:
            _UNI["status"] = "idle"      # stale flag from a dead/old run — clear it
    return alive


@app.route("/api/uni/start", methods=["POST"])
def uni_start():
    from .. import university
    p = request.get_json(force=True)
    model = (p.get("model") or "rl-graduate").strip()
    domain = (p.get("domain") or "").strip()
    if not domain:
        return jsonify({"error": "enter a domain to specialise in"}), 400
    if (_UNI["status"] == "running" and _uni_alive()) or _UNI["rebuild"] == "running":
        return jsonify({"error": "a run is already in progress", "can_force": True}), 409
    busy = _gpu_claim("University")
    if busy:
        return jsonify({"error": f"the GPU is busy ({busy}) — let it finish first", "can_force": True}), 409
    _UNI.update({"status": "running", "events": [], "summary": None, "stop": False, "log_path": None,
                 "active": {}, "active_model": model, "chat": [],
                 "control": {"guidance": [], "questions": [], "curriculum_ops": [], "overrides": {}}})

    n_students = _clamp(p.get("students", 1), 1, 4, 1)

    def run():
        try:
            student = (p.get("student") or "").strip()
            if student:
                from ..llm import engine
                engine.set_base_model(student)
            common = dict(
                target=_clamp(p.get("target", 0.7), 0.3, 1.0, 0.7),
                max_subjects=_clamp(p.get("max_subjects", 0), 0, 40, 0),
                max_minutes=_clamp(p.get("max_minutes", 20), 1.0, 240.0, 20.0),
                max_budget=_clamp(p.get("max_budget", 0.5), 0.0, 50.0, 0.5),
                depth=_clamp(p.get("depth", 18), 6, 120, 18),
                provider=_LLM["provider"], api_key=_LLM["api_key"],
                on_event=lambda ev: _UNI["events"].append(ev),
                on_node=lambda d: _UNI["active"].update(d), on_chat=_uni_chat,
                control=_UNI["control"], stop_flag=lambda: _UNI["stop"])
            if n_students > 1:
                s = university.run_cohort(model, domain, n_students=n_students, **common)
            else:
                s = university.run_university(model, domain, **common)
            _UNI["summary"] = s
            _UNI["log_path"] = s.get("log_path")
            _UNI["status"] = "done"
        except Exception as exc:
            _UNI["status"] = "error"
            _UNI["events"].append({"type": "error", "msg": repr(exc)})
        finally:
            _gpu_release("University")

    t = threading.Thread(target=run, daemon=True)
    _UNI["thread"] = t
    t.start()
    return jsonify({"ok": True})


@app.route("/api/uni/stop", methods=["POST"])
def uni_stop():
    _UNI["stop"] = True
    return jsonify({"ok": True})


@app.route("/api/uni/force_reset", methods=["POST"])
def uni_force_reset():
    """Killswitch: clear a stuck/old run so a new one can start. Signals any live
    worker to stop (it winds down at the next checkpoint) and resets the state."""
    _UNI["stop"] = True
    was_alive = _uni_alive()
    with _JOBS_LOCK:
        _UNI.update({"status": "idle", "rebuild": "idle", "active": {}})
        if not was_alive and _GPU["owner"] in ("University", "University rebuild"):
            _GPU["owner"] = None          # dead worker can't release its claim
    return jsonify({"ok": True, "was_alive": was_alive})


@app.route("/api/uni/intervene", methods=["POST"])
def uni_intervene():
    """Step into a live run: guide the teacher, ask the student, override a grade,
    or edit the curriculum. Queued into _UNI['control'] for the worker to pick up."""
    if not _uni_alive():
        return jsonify({"ok": False, "error": "no run in progress"})
    p = request.get_json(force=True)
    action = p.get("action")
    ctl = _UNI["control"]
    if action == "guide" and p.get("text"):
        ctl["guidance"].append(p["text"].strip())
    elif action == "ask" and p.get("text"):
        ctl["questions"].append(p["text"].strip())
    elif action == "override" and p.get("question") and p.get("verdict") in ("pass", "fail", "partial"):
        ctl["overrides"][p["question"]] = p["verdict"]
    elif action == "curriculum" and p.get("op") in ("add", "remove"):
        ctl["curriculum_ops"].append({"op": p["op"], "subject": (p.get("subject") or "").strip(),
                                      "why": p.get("why", "")})
    else:
        return jsonify({"ok": False, "error": "unknown or incomplete intervention"})
    return jsonify({"ok": True})


@app.route("/api/uni/status")
def uni_status():
    return jsonify({"status": _UNI["status"], "events": _UNI["events"], "summary": _UNI["summary"],
                    "active": _UNI.get("active", {}), "rebuild": _UNI.get("rebuild", "idle"),
                    "active_model": _UNI.get("active_model"), "has_log": bool(_UNI.get("log_path")),
                    "chat": _UNI.get("chat", [])[-150:]})


@app.route("/api/uni/activate", methods=["POST"])
def uni_activate():
    from .. import university
    model = request.get_json(force=True).get("model")
    if not model:
        return jsonify({"error": "no model"}), 400
    ok = _uni_rebuild_job(lambda: (university.rebuild_model(model), _UNI.update({"active_model": model})))
    return jsonify({"ok": ok})


@app.route("/api/uni/log")
def uni_log():
    lp = _UNI.get("log_path")
    if not lp or not os.path.exists(lp):
        return jsonify({"error": "no transcript yet"}), 404
    return send_from_directory(os.path.dirname(lp), os.path.basename(lp), as_attachment=True)


@app.route("/api/uni/models")
def uni_models():
    from .. import university
    return jsonify({"models": university.list_models()})


@app.route("/api/uni/manifest")
def uni_manifest():
    from .. import university
    return jsonify(university.load_manifest(request.args.get("model", "rl-graduate")))


@app.route("/api/uni/delete", methods=["POST"])
def uni_delete():
    from .. import university
    if _UNI["status"] == "running" or _UNI["rebuild"] == "running":
        return jsonify({"ok": False, "error": "busy — stop the current run first"})
    model = request.get_json(force=True).get("model", "")
    ok = university.delete_model(model)
    if _UNI.get("active_model") == model:
        _UNI["active_model"] = None
    return jsonify({"ok": ok})


@app.route("/api/uni/record")
def uni_record():
    """The model's full learning timeline: every lesson with its coursework and
    exam answers, in the order it was taught."""
    from .. import university
    model = request.args.get("model", "rl-graduate")
    lessons = university.load_lessons(model)
    return jsonify({"model": model, "lessons": lessons, "n": len(lessons)})


@app.route("/api/uni/ask", methods=["POST"])
def uni_ask():
    """Test the trained model. With library=true, the model's 'library card' is
    used: relevant coursework from its enabled skills is retrieved and given as
    context (weights hold competence; the library holds the facts)."""
    from .. import university
    from ..llm import engine
    p = request.get_json(force=True)
    q = (p.get("question") or "").strip()
    if not q:
        return jsonify({"error": "ask a question"}), 400
    model = p.get("model") or _UNI.get("active_model")
    notes = []
    if p.get("library") and model:
        notes = university.library_lookup(model, q)
    user = q
    if notes:
        ctx = "\n\n".join(f"[note {i + 1}] Q: {n['question']}\nA: {n['answer']}" for i, n in enumerate(notes))
        user = f"You may use these notes from your own coursework if relevant:\n{ctx}\n\nQuestion: {q}"
    try:
        base = engine.chat(engine.SYS, q, tuned=False)
        tuned = engine.chat(engine.SYS, user, tuned=True) if engine.status()["has_tuned"] else None
    except Exception as exc:
        return jsonify({"error": repr(exc)}), 500
    return jsonify({"base": base, "tuned": tuned,
                    "library_used": [n["question"] for n in notes]})


@app.route("/api/uni/theses")
def uni_theses():
    """The graduation thesis library — one entry per agent that graduated."""
    from .. import university
    return jsonify({"theses": university.list_theses()})


@app.route("/api/uni/thesis")
def uni_thesis():
    from .. import university
    rec = university.get_thesis(request.args.get("id", ""))
    if not rec:
        return jsonify({"error": "thesis not found"}), 404
    return jsonify(rec)


@app.route("/api/uni/thesis_download")
def uni_thesis_download():
    from .. import university
    tid = university._safe(request.args.get("id", ""))
    path = os.path.join(university.THESES_DIR, tid + ".md")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(university.THESES_DIR, tid + ".md", as_attachment=True)


def _uni_rebuild_job(fn):
    if _UNI["status"] == "running" or _UNI["rebuild"] == "running":
        return False
    if _gpu_claim("University rebuild"):
        return False
    with _JOBS_LOCK:
        _UNI["rebuild"] = "running"

    def run():
        try:
            fn()
        finally:
            _UNI["rebuild"] = "done"
            _gpu_release("University rebuild")
    threading.Thread(target=run, daemon=True).start()
    return True


@app.route("/api/uni/skill_toggle", methods=["POST"])
def uni_skill_toggle():
    from .. import university
    p = request.get_json(force=True)
    _UNI["active_model"] = p["model"]
    ok = _uni_rebuild_job(lambda: university.set_skill_enabled(p["model"], p["skill"], bool(p.get("enabled"))))
    return jsonify({"ok": ok})


@app.route("/api/uni/skill_retrain", methods=["POST"])
def uni_skill_retrain():
    from .. import university
    p = request.get_json(force=True)
    _UNI["active_model"] = p["model"]
    ok = _uni_rebuild_job(lambda: university.retrain_skill(p["model"], p["skill"], _LLM["provider"], _LLM["api_key"]))
    return jsonify({"ok": ok})


# (The old /api/auto/* Auto-Improve endpoints were removed: the University replaced
# that feature. auto_improve.py itself stays — LocalTeacher/make_teacher live there.)


# ---- Reasoning Lab: run a graph of prompted agents ----
_REASON = {"status": "idle", "steps": [], "final": "", "iterations": 0, "error": None}


@app.route("/api/reason/node_types")
def reason_node_types():
    from ..reasoning import NODE_TYPES
    return jsonify({"types": [
        {"id": k, "label": v["label"], "kind": v["kind"], "ports": v["ports"], "prompt": v.get("prompt", "")}
        for k, v in NODE_TYPES.items()
    ]})


@app.route("/api/reason/run", methods=["POST"])
def reason_run():
    from .. import reasoning
    p = request.get_json(force=True)
    graph, q = p.get("graph", {}), (p.get("question") or "").strip()
    if not q:
        return jsonify({"error": "enter a question"}), 400
    if _REASON["status"] == "running":
        return jsonify({"error": "a reasoning run is already in progress"}), 409
    _REASON.update({"status": "running", "steps": [], "final": "", "iterations": 0, "error": None})

    def run():
        try:
            res = reasoning.run_graph(graph, q, provider=_LLM["provider"], api_key=_LLM["api_key"],
                                      on_step=lambda s: _REASON["steps"].append(s))
            _REASON["final"] = res.get("final", "")
            _REASON["iterations"] = res.get("iterations", 0)
            if res.get("error"):
                _REASON["error"] = res["error"]
            _REASON["status"] = "done"
        except Exception as exc:
            _REASON["status"] = "error"
            _REASON["error"] = repr(exc)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/reason/status")
def reason_status():
    return jsonify(_REASON)


@app.route("/api/agent_code")
def agent_code():
    """The real source of an algorithm — shown by the 'see the code' button."""
    entry = GAME_AGENTS.get(request.args.get("id"))
    if entry is None:
        return jsonify({"error": "unknown agent"}), 404
    parts = []
    for obj in entry["code"]:
        try:
            parts.append(inspect.getsource(obj))
        except Exception as exc:
            parts.append(f"# (source unavailable: {exc})")
    return jsonify({"label": entry["label"], "code": "\n\n".join(parts)})


@app.route("/api/model_save", methods=["POST"])
def model_save():
    payload = request.get_json(force=True)
    job = _GAME_JOBS.get(payload.get("job_id"))
    if job is None or not job.get("net"):
        return jsonify({"error": "no trained model for that job (train one first)"}), 404
    name = persistence.save_model(payload.get("name") or "model", job["net"], job["config"], job["meta"])
    return jsonify({"ok": True, "name": name})


@app.route("/api/models")
def models_list():
    return jsonify({"models": persistence.list_models()})


@app.route("/api/model_test", methods=["POST"])
def model_test():
    """Load a saved model and play one greedy game; return frames to animate."""
    payload = request.get_json(force=True)
    loaded = persistence.load_model(payload.get("name", ""))
    if loaded is None:
        return jsonify({"error": "model not found"}), 404
    net, config, meta = loaded
    env = _game_cls(meta.get("game", "pacman"))()
    roll = persistence.play_record(env, net)
    return jsonify({"meta": meta, "layout": env.static_layout(), "roll": roll})


@app.route("/api/model_delete", methods=["POST"])
def model_delete():
    persistence.delete_model(request.get_json(force=True).get("name", ""))
    return jsonify({"ok": True})


def _banner(settings) -> None:
    # Deliberately ASCII: a Windows console is cp1252 by default and printing a
    # dash or an emoji here would raise UnicodeEncodeError before the server
    # ever starts.
    if settings.serves_ui:
        print(f"\n  RL Learning playground  ->  http://127.0.0.1:{settings.port}")
        print("  Running fully locally. No account, no cloud, no AWS.\n")
        return

    print("\n  RL Learning - GPU worker")
    print(f"  Listening on http://127.0.0.1:{settings.port} (this machine only)")
    print("\n  Pair the web app with this machine using:\n")
    print(f"      {settings.token}\n")
    if settings.allowed_origins:
        print(f"  Accepting the UI at: {', '.join(settings.allowed_origins)}")
    else:
        print("  WARNING: no --allow-origin set; the browser UI will be refused.")
    print("  Your GPU, models and API keys stay on this machine.\n")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="rl-playground", description=__doc__)
    parser.add_argument(
        "--mode", choices=[config.LOCAL, config.WORKER],
        help="local: run everything here, serve the UI, no AWS (default). "
             "worker: be the GPU engine for the hosted UI.",
    )
    parser.add_argument("--port", type=int, help="port to listen on")
    parser.add_argument(
        "--allow-origin", action="append", dest="allowed_origins", metavar="URL",
        help="cloud UI origin permitted to drive this worker (repeatable)",
    )
    parser.add_argument("--debug", action="store_true", default=None,
                        help="Flask auto-reloader (development only)")
    args = parser.parse_args(argv)

    settings = config.load(
        mode=args.mode, port=args.port, debug=args.debug,
        allowed_origins=args.allowed_origins,
    )
    # app-level hooks were installed at import time from the env; re-apply so
    # command-line flags (e.g. --mode worker) actually take effect.
    globals()["SETTINGS"] = settings
    security.init(app, settings)

    _banner(settings)

    if settings.debug:
        app.run(host=settings.host, port=settings.port, debug=True, threaded=True)
        return

    # Prefer a real WSGI server: training runs for minutes and the dev server is
    # not meant to be left running on someone's machine. But not having it is no
    # reason to refuse to start -- fall back and say so.
    try:
        from waitress import serve
    except ImportError:
        print("  (waitress not installed -- using the Flask dev server.")
        print("   For the real thing: pip install waitress)\n")
        app.run(host=settings.host, port=settings.port, threaded=True)
        return

    serve(app, host=settings.host, port=settings.port, threads=8)


if __name__ == "__main__":
    main()
