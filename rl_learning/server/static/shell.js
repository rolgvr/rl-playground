/* The learning-ladder shell.
 *
 * This is the platform's primary navigation. It replaces the old flat tab row
 * with a guided-but-free ladder: six rungs grouped into three acts that carry a
 * learner from search → reinforcement learning → agentic LLMs. Any rung is
 * clickable at any time (free jump); the recommended next one is highlighted.
 *
 * It drives the SAME views app.js already renders — shell.js owns navigation,
 * app.js owns each view's behaviour. The two meet at one function, window.setView.
 */
(function () {
  "use strict";

  // The ladder. Order == the intended journey. `view` maps to app.js's setView.
  // Every rung runs on the user's worker (compute is never in the cloud); the
  // `needsGpu` ones additionally require a CUDA GPU.
  const RUNGS = [
    { view: "grid",   n: 1, act: "Search",                 icon: "▦",  title: "Grid maze",     blurb: "Pathfinding on a maze you draw — BFS, Dijkstra, A*.", needsGpu: false },
    { view: "map",    n: 2, act: "Search",                 icon: "🗺", title: "Real streets",  blurb: "The very same searches, racing over live OpenStreetMap roads.", needsGpu: false },
    { view: "game",   n: 3, act: "Reinforcement Learning", icon: "🎮", title: "Deep RL",       blurb: "An agent learns Pac-Man & Pong from raw pixels — DQN to PPO.", needsGpu: true },
    { view: "llm",    n: 4, act: "Agentic LLMs",           icon: "🧠", title: "Teach an LLM",  blurb: "Distil a big teacher into a small model (SFT → DPO / RLAIF).", needsGpu: true },
    { view: "reason", n: 5, act: "Agentic LLMs",           icon: "🔁", title: "Reasoning Lab", blurb: "Wire agents into generate → critique → refine loops.", needsGpu: true },
    { view: "auto",   n: 6, act: "Agentic LLMs",           icon: "🎓", title: "University",     blurb: "A Dean designs a curriculum and teaches a model to mastery.", needsGpu: true },
  ];
  const ACTS = ["Search", "Reinforcement Learning", "Agentic LLMs"];
  const byView = Object.fromEntries(RUNGS.map((r) => [r.view, r]));

  // Which rungs the learner has opened — powers the ✓ marks and "recommended next".
  const VISITED = new Set(JSON.parse(localStorage.getItem("rl_visited") || "[]"));
  const saveVisited = () => localStorage.setItem("rl_visited", JSON.stringify([...VISITED]));

  let current = "grid";

  // Is a worker attached at all? Every rung needs this (compute runs there).
  function connected() {
    return !!(window.RL && window.RL.paired);
  }
  function gpuReady() {
    const caps = window.RL && window.RL.capabilities;
    return !!(connected() && caps && caps.gpu && caps.gpu.available);
  }
  // A rung is usable when its requirements are met.
  function satisfied(r) {
    return connected() && (!r.needsGpu || gpuReady());
  }

  // The first not-yet-visited rung: what we nudge the learner toward next.
  function recommendedView() {
    const next = RUNGS.find((r) => !VISITED.has(r.view));
    return next ? next.view : null;
  }

  function statusChip(r) {
    if (r.needsGpu && !gpuReady()) return { cls: "chip-gpu", text: "needs GPU" };
    if (!r.needsGpu && !connected()) return { cls: "chip-worker", text: "needs worker" };
    if (VISITED.has(r.view)) return { cls: "chip-done", text: "✓" };
    return null;
  }

  function renderLadder() {
    const root = document.getElementById("ladder");
    if (!root) return;
    const rec = recommendedView();

    let html = "";
    for (const act of ACTS) {
      html += `<div class="act-label">${act}</div>`;
      for (const r of RUNGS.filter((x) => x.act === act)) {
        const chip = statusChip(r);
        const cls = [
          "ladder-rung",
          r.view === current ? "is-current" : "",
          r.view === rec && r.view !== current ? "is-next" : "",
          VISITED.has(r.view) ? "is-visited" : "",
        ].join(" ").trim();
        html += `
          <button class="${cls}" data-rung="${r.view}">
            <span class="rung-n">${r.n}</span>
            <span class="rung-main">
              <span class="rung-title">${r.icon} ${r.title}</span>
              <span class="rung-blurb">${r.blurb}</span>
            </span>
            ${chip ? `<span class="rung-chip ${chip.cls}">${chip.text}</span>` : ""}
            ${r.view === rec && r.view !== current ? `<span class="rung-chip chip-next">next</span>` : ""}
          </button>`;
      }
    }
    root.innerHTML = html;

    root.querySelectorAll("[data-rung]").forEach((btn) => {
      btn.addEventListener("click", () => go(btn.dataset.rung));
    });
  }

  function renderHeader() {
    const el = document.getElementById("rungHeader");
    const r = byView[current];
    if (!el || !r) return;

    // Requirement badge: GPU rungs need a CUDA worker; the rest need any worker.
    let badge;
    if (r.needsGpu) {
      badge = gpuReady()
        ? `<span class="rh-gpu ok">⚡ GPU worker</span>`
        : `<span class="rh-gpu locked">🔒 needs a GPU worker</span>`;
    } else {
      badge = connected()
        ? `<span class="rh-free">⚙ running on your worker</span>`
        : `<span class="rh-gpu locked">🔌 needs the worker</span>`;
    }

    // A call-to-action when the current rung can't run yet.
    let gate = "";
    if (!satisfied(r)) {
      let msg, showConnect = true;
      if (!connected()) {
        msg = r.needsGpu
          ? "This rung trains on a GPU on your own machine. Connect your worker to use it."
          : "This platform runs on your own machine — nothing computes in the cloud. Connect your worker to use this rung.";
      } else {
        // Connected, but this GPU rung has no CUDA GPU on the worker.
        msg = "This rung needs a CUDA GPU, and the connected worker doesn’t have one.";
        showConnect = false;
      }
      gate = `
        <div class="connect-gate">
          <span>${msg}</span>
          ${showConnect ? `<button class="worker-btn worker-btn-primary" id="gateConnectBtn">Connect the worker</button>` : ""}
        </div>`;
    }

    el.innerHTML = `
      <div class="rh-top">
        <span class="rh-n">Rung ${r.n} / ${RUNGS.length}</span>
        <span class="rh-act">${r.act}</span>
        ${badge}
      </div>
      <h1 class="rh-title">${r.icon} ${r.title}</h1>
      <p class="rh-blurb">${r.blurb}</p>
      ${gate}`;

    const gc = document.getElementById("gateConnectBtn");
    if (gc) gc.addEventListener("click", () => {
      const btn = document.getElementById("workerConnectBtn");
      if (btn) btn.click();          // reuse connect.js's wired pairing dialog
    });
  }

  // The one navigation entry point. Delegates the actual view swap to app.js.
  function go(view) {
    if (!byView[view]) return;
    current = view;
    VISITED.add(view);
    saveVisited();
    if (typeof window.setView === "function") window.setView(view);
    renderLadder();
    renderHeader();
  }
  window.goRung = go;

  function refresh() {
    renderLadder();
    renderHeader();
    // The algorithm list is served by the worker; without one it can't load, so
    // say so instead of leaving an empty box. app.js overwrites this once paired.
    const algo = document.getElementById("algoList");
    if (algo && !connected() && !algo.querySelector(".algo-item")) {
      algo.innerHTML = `<p class="muted tiny-note">Connect the worker to load the algorithms.</p>`;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    // app.js boots on the grid view; the ladder starts there too.
    current = "grid";
    VISITED.add(current);
    saveVisited();
    refresh();
  });

  // Chips flip from "needs GPU" to "⚡ GPU" the moment a worker with a GPU pairs.
  document.addEventListener("rl-worker-ready", refresh);
})();
