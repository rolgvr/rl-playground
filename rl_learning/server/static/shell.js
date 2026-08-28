/* The platform shell: catalog-driven navigation.
 *
 * Renders the playground rail from window.RL_TRACKS (tracks.js) — the live
 * track's modules and lessons, plus the tracks still in development. Any
 * lesson is clickable at any time (free jump); the recommended next one is
 * highlighted.
 *
 * It drives the SAME views app.js already renders — shell.js owns navigation,
 * app.js owns each view's behaviour. The two meet at one function, window.setView.
 */
(function () {
  "use strict";

  const TRACKS = window.RL_TRACKS || [];
  const TRACK = TRACKS.find((t) => t.status === "live") || { modules: [] };
  const SOON = TRACKS.filter((t) => t.status !== "live");

  // Flat lesson list in journey order; `view` maps to app.js's setView.
  const LESSONS = TRACK.modules.flatMap((m) =>
    m.lessons.map((l) => ({ ...l, act: m.title })));
  const byView = Object.fromEntries(LESSONS.map((l) => [l.view, l]));
  const ic = (name, cls) => (window.icon ? window.icon(name, cls) : "");

  // Which lessons the learner has opened — powers the ✓ marks and "next".
  const VISITED = new Set(JSON.parse(localStorage.getItem("rl_visited") || "[]"));
  const saveVisited = () => localStorage.setItem("rl_visited", JSON.stringify([...VISITED]));

  let current = "grid";

  // Is a worker attached at all? Every lesson needs this (compute runs there).
  function connected() {
    return !!(window.RL && window.RL.paired);
  }
  function gpuReady() {
    const caps = window.RL && window.RL.capabilities;
    return !!(connected() && caps && caps.gpu && caps.gpu.available);
  }
  // A lesson is usable when its requirements are met.
  function satisfied(l) {
    return connected() && (!l.needsGpu || gpuReady());
  }

  // The first not-yet-visited lesson: what we nudge the learner toward next.
  function recommendedView() {
    const next = LESSONS.find((l) => !VISITED.has(l.view));
    return next ? next.view : null;
  }

  function statusChip(l) {
    if (l.needsGpu && !gpuReady()) return { cls: "chip-gpu", text: "needs GPU" };
    if (!l.needsGpu && !connected()) return { cls: "chip-worker", text: "needs worker" };
    if (VISITED.has(l.view)) return { cls: "chip-done", text: "✓" };
    return null;
  }

  function renderLadder() {
    const root = document.getElementById("ladder");
    if (!root) return;
    const rec = recommendedView();

    let html = `<div class="rail-track">${ic(TRACK.icon)} ${TRACK.title}</div>`;
    for (const mod of TRACK.modules) {
      html += `<div class="act-label">${mod.title}</div>`;
      for (const l of mod.lessons) {
        const chip = statusChip(l);
        const cls = [
          "ladder-rung",
          l.view === current ? "is-current" : "",
          l.view === rec && l.view !== current ? "is-next" : "",
          VISITED.has(l.view) ? "is-visited" : "",
        ].join(" ").trim();
        html += `
          <button class="${cls}" data-rung="${l.view}">
            <span class="rung-n">${l.n}</span>
            <span class="rung-main">
              <span class="rung-title">${ic(l.icon)} ${l.title}</span>
              <span class="rung-blurb">${l.blurb}</span>
            </span>
            ${chip ? `<span class="rung-chip ${chip.cls}">${chip.text}</span>` : ""}
            ${l.view === rec && l.view !== current ? `<span class="rung-chip chip-next">next</span>` : ""}
          </button>`;
      }
    }

    if (SOON.length) {
      html += `<div class="rail-soon"><div class="act-label">More tracks</div>`;
      for (const t of SOON) {
        html += `
          <div class="rail-soon-row" title="${t.tagline || ""}">
            ${ic(t.icon)} ${t.title}
            <span class="badge-soon">in development</span>
          </div>`;
      }
      html += `</div>`;
    }
    root.innerHTML = html;

    root.querySelectorAll("[data-rung]").forEach((btn) => {
      btn.addEventListener("click", () => go(btn.dataset.rung));
    });
  }

  function renderHeader() {
    const el = document.getElementById("rungHeader");
    const l = byView[current];
    if (!el || !l) return;

    // Requirement badge: GPU lessons need a CUDA worker; the rest need any worker.
    let badge;
    if (l.needsGpu) {
      badge = gpuReady()
        ? `<span class="rh-gpu ok">${ic("bolt")} GPU worker</span>`
        : `<span class="rh-gpu locked">${ic("lock")} needs a GPU worker</span>`;
    } else {
      badge = connected()
        ? `<span class="rh-free">${ic("check")} running on your worker</span>`
        : `<span class="rh-gpu locked">${ic("plug")} needs the worker</span>`;
    }

    // A call-to-action when the current lesson can't run yet.
    let gate = "";
    if (!satisfied(l)) {
      let msg, showConnect = true;
      if (!connected()) {
        msg = l.needsGpu
          ? "This lesson trains on a GPU on your own machine. Connect your worker to use it."
          : "This platform runs on your own machine — nothing computes in the cloud. Connect your worker to use this lesson.";
      } else {
        // Connected, but this GPU lesson has no CUDA GPU on the worker.
        msg = "This lesson needs a CUDA GPU, and the connected worker doesn’t have one.";
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
        <span class="rh-n">Lesson ${l.n} / ${LESSONS.length}</span>
        <span class="rh-act">${TRACK.title} · ${l.act}</span>
        ${badge}
      </div>
      <h1 class="rh-title">${ic(l.icon)} ${l.title}</h1>
      <p class="rh-blurb">${l.blurb} <a href="${l.learn}">Theory →</a></p>
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
    // app.js boots on the grid view; the rail starts there too.
    current = "grid";
    VISITED.add(current);
    saveVisited();
    refresh();
  });

  // Chips flip from "needs GPU" to "GPU worker" the moment a worker pairs.
  document.addEventListener("rl-worker-ready", refresh);
})();
