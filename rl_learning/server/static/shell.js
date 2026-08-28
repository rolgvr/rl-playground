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
  const RUNGS = [
    { view: "grid",   n: 1, act: "Search",                 icon: "▦",  title: "Grid maze",     blurb: "Pathfinding on a maze you draw — BFS, Dijkstra, A*.", gpu: false },
    { view: "map",    n: 2, act: "Search",                 icon: "🗺", title: "Real streets",  blurb: "The very same searches, racing over live OpenStreetMap roads.", gpu: false },
    { view: "game",   n: 3, act: "Reinforcement Learning", icon: "🎮", title: "Deep RL",       blurb: "An agent learns Pac-Man & Pong from raw pixels — DQN to PPO.", gpu: true },
    { view: "llm",    n: 4, act: "Agentic LLMs",           icon: "🧠", title: "Teach an LLM",  blurb: "Distil a big teacher into a small model (SFT → DPO / RLAIF).", gpu: true },
    { view: "reason", n: 5, act: "Agentic LLMs",           icon: "🔁", title: "Reasoning Lab", blurb: "Wire agents into generate → critique → refine loops.", gpu: true },
    { view: "auto",   n: 6, act: "Agentic LLMs",           icon: "🎓", title: "University",     blurb: "A Dean designs a curriculum and teaches a model to mastery.", gpu: true },
  ];
  const ACTS = ["Search", "Reinforcement Learning", "Agentic LLMs"];
  const byView = Object.fromEntries(RUNGS.map((r) => [r.view, r]));

  // Which rungs the learner has opened — powers the ✓ marks and "recommended next".
  const VISITED = new Set(JSON.parse(localStorage.getItem("rl_visited") || "[]"));
  const saveVisited = () => localStorage.setItem("rl_visited", JSON.stringify([...VISITED]));

  let current = "grid";

  function gpuReady() {
    const caps = window.RL && window.RL.capabilities;
    return !!(window.RL && window.RL.paired && caps && caps.gpu && caps.gpu.available);
  }

  // The first not-yet-visited rung: what we nudge the learner toward next.
  function recommendedView() {
    const next = RUNGS.find((r) => !VISITED.has(r.view));
    return next ? next.view : null;
  }

  function statusChip(r) {
    if (r.gpu && !gpuReady()) return { cls: "chip-gpu", text: "needs GPU" };
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
    const locked = r.gpu && !gpuReady();
    el.innerHTML = `
      <div class="rh-top">
        <span class="rh-n">Rung ${r.n} / ${RUNGS.length}</span>
        <span class="rh-act">${r.act}</span>
        ${r.gpu ? `<span class="rh-gpu ${locked ? "locked" : "ok"}">${locked ? "🔒 needs a GPU worker" : "⚡ GPU"}</span>` : `<span class="rh-free">free · in browser</span>`}
      </div>
      <h1 class="rh-title">${r.icon} ${r.title}</h1>
      <p class="rh-blurb">${r.blurb}</p>`;
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
