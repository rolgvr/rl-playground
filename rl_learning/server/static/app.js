/* RL Learning playground — Stage 1 frontend.
 *
 * Responsibilities, kept deliberately small:
 *   1. Let the user paint ONE grid (walls, mud, start, goal).
 *   2. POST that grid to /api/solve and get back a Trace per algorithm.
 *   3. Replay every Trace in lockstep so the user watches the algorithms race.
 *
 * The frontend knows nothing about *how* an algorithm works — it just animates
 * the `visited` list and draws the `path`. That is why the same code will later
 * animate a Q-learning agent without changes.
 */

const ROWS = 20;
const COLS = 30;
const EDITOR_CELL = 24;
const MUD_WEIGHT = 5;

// Per-algorithm colour so each card is identifiable at a glance.
const ALGO_COLOR = {
  bfs: "#5b8cff",
  dfs: "#c77dff",
  dijkstra: "#3ad29f",
  astar: "#ffd166",
  greedy: "#ff9f45",
  weighted_astar: "#f25f5c",
  bi_bfs: "#4cc9f0",
  bi_dijkstra: "#80ed99",
  bi_astar: "#e0aaff",
};

// ------------------------------------------------------------- theme bridge
// Canvas 2D can't use var(--x), so the world/curve renderers resolve the SAME
// CSS tokens the stylesheet uses via RLTheme (theme.js). T() keeps a fallback
// so app.js still renders if theme.js is somehow missing.
const T = (token, fallback) => (window.RLTheme && RLTheme.get(token)) || fallback;
// When the theme flips, re-render the theme-following canvases. Animated
// surfaces (races, curves, games) pick the new tokens up on their next frame.
document.addEventListener("rl-theme-change", () => { redrawEditor(); });

// ---------------------------------------------------------------- state
const state = {
  rows: ROWS,
  cols: COLS,
  walls: new Set(),           // "r,c"
  weights: new Map(),         // "r,c" -> number
  start: [Math.floor(ROWS / 2), 2],
  goal: [Math.floor(ROWS / 2), COLS - 3],
  diagonal: false,
  mode: "wall",
  selected: new Set(["bfs", "dijkstra", "astar"]),
  algorithms: [],             // metadata from server
  animating: false,
  rafId: null,

  // --- map view ---
  view: "grid",               // "grid" | "map"
  road: null,                 // { graph_id, nodes:{id:[lat,lon]}, edges, start, goal, bounds }
  mapPoints: [],              // up to two [lat,lon] clicks
};

const key = (r, c) => `${r},${c}`;
const parseKey = (k) => k.split(",").map(Number);
// A trace node is a grid cell [r,c] or a road-node id (number). This gives each
// a stable Set key so the same race-animation code works in both views.
const nodeKey = (n) => (Array.isArray(n) ? `${n[0]},${n[1]}` : n);

// ---------------------------------------------------------------- editor canvas
const editor = document.getElementById("editor");
const ectx = editor.getContext("2d");
editor.width = COLS * EDITOR_CELL;
editor.height = ROWS * EDITOR_CELL;

function drawGrid(ctx, cell, opts = {}) {
  const { visited, path, visitedColor = T("--visited", "#2a4a7a"), pathColor = T("--path", "#ffd166") } = opts;
  const { rows, cols, walls, weights, start, goal } = state;

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cell, y = r * cell;
      const k = key(r, c);
      let fill = T("--cell", "#11141b");
      if (walls.has(k)) fill = T("--wall", "#2b2f3c");
      else if (weights.has(k)) fill = T("--mud", "#6b4f2a");
      ctx.fillStyle = fill;
      ctx.fillRect(x, y, cell, cell);
    }
  }

  // explored cells (drawn translucent over terrain)
  if (visited && visited.size) {
    ctx.fillStyle = visitedColor;
    ctx.globalAlpha = 0.55;
    for (const k of visited) {
      const [r, c] = parseKey(k);
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
    ctx.globalAlpha = 1;
  }

  // grid lines
  ctx.strokeStyle = T("--grid-line", "#262b38");
  ctx.lineWidth = 1;
  for (let r = 0; r <= rows; r++) {
    ctx.beginPath(); ctx.moveTo(0, r * cell); ctx.lineTo(cols * cell, r * cell); ctx.stroke();
  }
  for (let c = 0; c <= cols; c++) {
    ctx.beginPath(); ctx.moveTo(c * cell, 0); ctx.lineTo(c * cell, rows * cell); ctx.stroke();
  }

  // path
  if (path && path.length > 1) {
    ctx.strokeStyle = pathColor;
    ctx.lineWidth = Math.max(2, cell * 0.28);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    path.forEach(([r, c], i) => {
      const px = c * cell + cell / 2, py = r * cell + cell / 2;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  // start / goal markers
  drawMarker(ctx, cell, start, T("--start", "#3ad29f"));
  drawMarker(ctx, cell, goal, T("--goal", "#ff5c7a"));
}

function drawMarker(ctx, cell, [r, c], color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(c * cell + cell / 2, r * cell + cell / 2, cell * 0.3, 0, Math.PI * 2);
  ctx.fill();
}

function redrawEditor() { drawGrid(ectx, EDITOR_CELL); }

// ---------------------------------------------------------------- painting
let painting = false;

function cellFromEvent(e) {
  const rect = editor.getBoundingClientRect();
  const scaleX = editor.width / rect.width;
  const scaleY = editor.height / rect.height;
  const c = Math.floor(((e.clientX - rect.left) * scaleX) / EDITOR_CELL);
  const r = Math.floor(((e.clientY - rect.top) * scaleY) / EDITOR_CELL);
  if (r < 0 || r >= state.rows || c < 0 || c >= state.cols) return null;
  return [r, c];
}

function applyMode(r, c) {
  const k = key(r, c);
  const isStart = state.start[0] === r && state.start[1] === c;
  const isGoal = state.goal[0] === r && state.goal[1] === c;

  switch (state.mode) {
    case "wall":
      if (isStart || isGoal) return;
      state.weights.delete(k);
      state.walls.add(k);
      break;
    case "weight":
      if (isStart || isGoal) return;
      state.walls.delete(k);
      state.weights.set(k, MUD_WEIGHT);
      break;
    case "erase":
      state.walls.delete(k);
      state.weights.delete(k);
      break;
    case "start":
      if (isGoal || state.walls.has(k)) return;
      state.start = [r, c];
      break;
    case "goal":
      if (isStart || state.walls.has(k)) return;
      state.goal = [r, c];
      break;
  }
  redrawEditor();
}

editor.addEventListener("mousedown", (e) => {
  const cell = cellFromEvent(e);
  if (!cell) return;
  painting = true;
  applyMode(...cell);
});
editor.addEventListener("mousemove", (e) => {
  if (!painting) return;
  // start/goal are single-placement; don't smear them while dragging
  if (state.mode === "start" || state.mode === "goal") return;
  const cell = cellFromEvent(e);
  if (cell) applyMode(...cell);
});
window.addEventListener("mouseup", () => { painting = false; });

// ---------------------------------------------------------------- toolbar
document.getElementById("modeButtons").addEventListener("click", (e) => {
  const btn = e.target.closest(".tool");
  if (!btn) return;
  state.mode = btn.dataset.mode;
  document.querySelectorAll("#modeButtons .tool").forEach((b) => b.classList.toggle("active", b === btn));
});

document.getElementById("clearBtn").onclick = () => {
  state.walls.clear();
  state.weights.clear();
  redrawEditor();
};

document.getElementById("resetBtn").onclick = () => {
  state.walls.clear();
  state.weights.clear();
  state.start = [Math.floor(ROWS / 2), 2];
  state.goal = [Math.floor(ROWS / 2), COLS - 3];
  redrawEditor();
};

document.getElementById("diagonalToggle").onchange = (e) => {
  state.diagonal = e.target.checked;
};

document.getElementById("mazeBtn").onclick = () => {
  // Sparse random obstacles — dense enough to make algorithms diverge,
  // sparse enough to almost always leave the goal reachable.
  state.walls.clear();
  state.weights.clear();
  for (let r = 0; r < state.rows; r++) {
    for (let c = 0; c < state.cols; c++) {
      const isEndpoint =
        (state.start[0] === r && state.start[1] === c) ||
        (state.goal[0] === r && state.goal[1] === c);
      if (isEndpoint) continue;
      const roll = Math.random();
      if (roll < 0.22) state.walls.add(key(r, c));
      else if (roll < 0.30) state.weights.set(key(r, c), MUD_WEIGHT);
    }
  }
  redrawEditor();
};

// ---------------------------------------------------------------- map view
let leaflet = null;            // the Leaflet map instance (created lazily)
let mapMarkers = [];           // start/goal pins
let networkLayer = null;       // drawn road graph
let pathLayers = [];           // result polylines on the big map

// The old flat tab row is gone (the shell rail is the navigation); the lookup
// stays guarded so this file also runs on pages without it.
const viewToggle = document.getElementById("viewToggle");
if (viewToggle) viewToggle.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  setView(btn.dataset.view);
});

function setView(view) {
  state.view = view;
  document.querySelectorAll("#viewToggle button").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view));
  document.getElementById("gridView").hidden = view !== "grid";
  document.getElementById("mapView").hidden = view !== "map";
  document.getElementById("gameView").hidden = view !== "game";
  document.getElementById("llmView").hidden = view !== "llm";
  document.getElementById("reasonView").hidden = view !== "reason";
  document.getElementById("autoView").hidden = view !== "auto";
  document.getElementById("pathControls").hidden = !(view === "grid" || view === "map");
  document.getElementById("gameControls").hidden = view !== "game";
  document.getElementById("llmControls").hidden = view !== "llm";
  document.getElementById("reasonControls").hidden = view !== "reason";
  document.getElementById("autoControls").hidden = view !== "auto";
  document.getElementById("llmFlow").hidden = view !== "llm";
  track.innerHTML = "";        // results from the other mode don't apply
  statusEl.textContent = "";
  if (view === "map") initMap();
  if (view === "game") initGame();
  if (view === "llm") initLlm();
  if (view === "reason") initReason();
  if (view === "auto") initAuto();
}

function initMap() {
  if (leaflet) { setTimeout(() => leaflet.invalidateSize(), 50); return; }
  leaflet = L.map("leafletMap").setView([40.0170, -105.2810], 15); // Boulder, CO
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(leaflet);
  leaflet.on("click", onMapClick);
}

function onMapClick(e) {
  const pt = [e.latlng.lat, e.latlng.lng];
  if (state.mapPoints.length >= 2) clearMapPoints();
  state.mapPoints.push(pt);
  const isStart = state.mapPoints.length === 1;
  const marker = L.circleMarker(e.latlng, {
    radius: 8, color: "#0b0e15", weight: 2,
    fillColor: isStart ? "#3ad29f" : "#ff5c7a", fillOpacity: 1,
  }).addTo(leaflet).bindTooltip(isStart ? "Start" : "Destination");
  mapMarkers.push(marker);
  document.getElementById("mapStatus").textContent =
    state.mapPoints.length === 1 ? "Now click a destination nearby." : "Ready — fetch the road network.";
}

function clearMapPoints() {
  state.mapPoints = [];
  mapMarkers.forEach((m) => leaflet.removeLayer(m));
  mapMarkers = [];
}

document.getElementById("mapClearBtn").onclick = () => {
  clearMapPoints();
  if (networkLayer) { leaflet.removeLayer(networkLayer); networkLayer = null; }
  pathLayers.forEach((l) => leaflet.removeLayer(l)); pathLayers = [];
  state.road = null;
  track.innerHTML = "";
  document.getElementById("mapStatus").textContent = "Click a start point on the map.";
};

document.getElementById("fetchRoadsBtn").onclick = async () => {
  if (state.mapPoints.length < 2) {
    document.getElementById("mapStatus").textContent = "Click two points first (start, then destination).";
    return;
  }
  const status = document.getElementById("mapStatus");
  status.textContent = "Fetching real roads from OpenStreetMap… (can take 10–30s)";

  let data;
  try {
    const res = await api("/api/roads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: state.mapPoints }),
    });
    data = await res.json();
    if (!res.ok) { status.textContent = "⚠ " + (data.error || "fetch failed"); return; }
  } catch (err) {
    status.textContent = "⚠ Network error reaching the server.";
    return;
  }

  state.road = data;
  drawNetworkOnMap(data);
  status.textContent = `Loaded ${data.counts.nodes} intersections, ${data.counts.edges} road segments. Press “Race selected”.`;
};

function drawNetworkOnMap(road) {
  if (networkLayer) leaflet.removeLayer(networkLayer);
  pathLayers.forEach((l) => leaflet.removeLayer(l)); pathLayers = [];
  const lines = road.edges.map(([a, b]) => [road.nodes[a], road.nodes[b]]);
  networkLayer = L.polyline(lines, { color: "#5b8cff", weight: 1.5, opacity: 0.5 }).addTo(leaflet);
  // snap markers to the actual graph nodes chosen as start/goal
  const s = road.nodes[road.start], g = road.nodes[road.goal];
  if (s && g) leaflet.fitBounds(L.latLngBounds([s, g]).pad(0.4));
}

// ---------------------------------------------------------------- algorithm picker
async function loadAlgorithms() {
  const res = await api("/api/algorithms");
  const data = await res.json();
  state.algorithms = data.algorithms;

  const list = document.getElementById("algoList");
  list.innerHTML = "";
  for (const algo of state.algorithms) {
    const item = document.createElement("div");
    item.className = "algo-item" + (state.selected.has(algo.id) ? " selected" : "");
    item.dataset.id = algo.id;
    item.innerHTML = `
      <div class="row">
        <span class="swatch" style="background:${ALGO_COLOR[algo.id] || "#888"}"></span>
        <b>${algo.label}</b>
        <span class="fam">${algo.family}</span>
      </div>
      <p>${algo.description}</p>`;
    item.onclick = () => {
      if (state.selected.has(algo.id)) state.selected.delete(algo.id);
      else state.selected.add(algo.id);
      item.classList.toggle("selected");
    };
    list.appendChild(item);
  }
}

function setAllSelected(on) {
  state.selected = new Set(on ? state.algorithms.map((a) => a.id) : []);
  document.querySelectorAll(".algo-item").forEach((el) =>
    el.classList.toggle("selected", state.selected.has(el.dataset.id))
  );
}
document.getElementById("selectAll").onclick = () => setAllSelected(true);
document.getElementById("selectNone").onclick = () => setAllSelected(false);

// ---------------------------------------------------------------- race
const statusEl = document.getElementById("status");
const raceBtn = document.getElementById("raceBtn");
const stopBtn = document.getElementById("stopBtn");
const speedEl = document.getElementById("speed");
const track = document.getElementById("raceTrack");

function gridPayload() {
  return {
    rows: state.rows,
    cols: state.cols,
    start: state.start,
    goal: state.goal,
    diagonal: state.diagonal,
    walls: [...state.walls].map(parseKey),
    weights: [...state.weights.entries()].map(([k, v]) => [parseKey(k), v]),
  };
}

raceBtn.onclick = async () => {
  if (state.animating) return;
  const chosen = state.algorithms.filter((a) => state.selected.has(a.id));
  if (!chosen.length) {
    statusEl.textContent = "Pick at least one algorithm to race.";
    return;
  }
  if (state.view === "map" && !state.road) {
    statusEl.textContent = "Fetch a road network first (Real streets view).";
    return;
  }

  statusEl.textContent = "Thinking…";
  raceBtn.disabled = true;

  let results;
  try {
    const endpoint = state.view === "map" ? "/api/solve_roads" : "/api/solve";
    const body = state.view === "map"
      ? { graph_id: state.road.graph_id, algorithms: chosen.map((a) => a.id) }
      : { grid: gridPayload(), algorithms: chosen.map((a) => a.id) };
    const res = await api(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    results = (await res.json()).results;
  } catch (err) {
    statusEl.textContent = "Server error — is the Flask app running?";
    raceBtn.disabled = false;
    return;
  }

  runRace(chosen, results);
};

stopBtn.onclick = () => stopRace();

function stopRace() {
  if (state.rafId) cancelAnimationFrame(state.rafId);
  state.animating = false;
  state.rafId = null;
  raceBtn.disabled = false;
  stopBtn.disabled = true;
}

// Build a lat/lon -> canvas projection that preserves aspect ratio (longitude
// is squeezed by cos(latitude)) and centres the network with a little padding.
function makeProjector(bounds, w, h, pad = 8) {
  const [s, west, n, east] = bounds;
  const cosL = Math.cos(((s + n) / 2) * Math.PI / 180);
  const gw = (east - west) * cosL, gh = (n - s);
  const scale = Math.min((w - 2 * pad) / gw, (h - 2 * pad) / gh);
  const offX = (w - gw * scale) / 2, offY = (h - gh * scale) / 2;
  return (lat, lon) => [offX + (lon - west) * cosL * scale, offY + (n - lat) * scale];
}

// Pre-render the faint base road network once per card (it never changes during
// the animation), so each frame only redraws the growing explored set + path.
function renderRoadBase(canvas, road, project) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = T("--cell", "#11141b");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = T("--road-base", "#33384a");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [a, b] of road.edges) {
    const A = road.nodes[a], B = road.nodes[b];
    if (!A || !B) continue;
    const [ax, ay] = project(A[0], A[1]);
    const [bx, by] = project(B[0], B[1]);
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
  }
  ctx.stroke();
}

function drawRoadCard(c, showPath) {
  const { ctx, base, road, project } = c;
  ctx.drawImage(base, 0, 0);

  // explored nodes as small dots
  ctx.fillStyle = ALGO_COLOR[c.algo.id] || "#5b8cff";
  for (const id of c.visited) {
    const p = road.nodes[id];
    if (!p) continue;
    const [x, y] = project(p[0], p[1]);
    ctx.fillRect(x - 1, y - 1, 2.5, 2.5);
  }

  if (showPath && c.trace.path.length > 1) {
    ctx.strokeStyle = T("--path-final", "#ffffff");
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    c.trace.path.forEach((id, i) => {
      const p = road.nodes[id]; if (!p) return;
      const [x, y] = project(p[0], p[1]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // start / goal
  for (const [id, col] of [[road.start, T("--start", "#3ad29f")], [road.goal, T("--goal", "#ff5c7a")]]) {
    const p = road.nodes[id]; if (!p) continue;
    const [x, y] = project(p[0], p[1]);
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
  }
}

function buildCards(chosen, results) {
  track.innerHTML = "";
  const cards = {};
  const onMap = state.view === "map";
  const cardCell = Math.max(8, Math.floor(248 / state.cols));
  const W = onMap ? 248 : state.cols * cardCell;
  const H = onMap ? 200 : state.rows * cardCell;
  const project = onMap ? makeProjector(state.road.bounds, W, H) : null;

  for (const algo of chosen) {
    const trace = results[algo.id];
    const card = document.createElement("div");
    card.className = "race-card";
    const costLabel = onMap ? "Distance" : "Path cost";
    card.innerHTML = `
      <div class="card-head">
        <div class="title">
          <span class="swatch" style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${ALGO_COLOR[algo.id] || "#888"}"></span>
          ${algo.label}
        </div>
        <span class="badge">running…</span>
      </div>
      <div class="canvas-host"><canvas></canvas></div>
      <div class="metrics">
        <div class="m"><span>Nodes expanded</span><b class="exp">0</b></div>
        <div class="m"><span>${onMap ? "Hops" : "Path length"}</span><b class="len">—</b></div>
        <div class="m"><span>${costLabel}</span><b class="cost">—</b></div>
        <div class="m"><span>Time</span><b class="time">—</b></div>
      </div>`;
    track.appendChild(card);

    const canvas = card.querySelector("canvas");
    canvas.width = W;
    canvas.height = H;

    const c = {
      algo, trace, card, canvas,
      ctx: canvas.getContext("2d"),
      cell: cardCell,
      visited: new Set(),
      done: false,
    };
    if (onMap) {
      c.road = state.road;
      c.project = project;
      c.base = document.createElement("canvas");
      c.base.width = W; c.base.height = H;
      renderRoadBase(c.base, state.road, project);
      drawRoadCard(c, false);          // show the empty network immediately
    }
    cards[algo.id] = c;
  }
  return cards;
}

function drawCard(c, showPath) {
  if (state.view === "map") { drawRoadCard(c, showPath); return; }
  drawGrid(c.ctx, c.cell, {
    visited: c.visited,
    path: showPath ? c.trace.path.map(([r, col]) => [r, col]) : null,
    visitedColor: ALGO_COLOR[c.algo.id] || T("--visited", "#2a4a7a"),
    pathColor: T("--path-final", "#ffffff"),
  });
}

function runRace(chosen, results) {
  stopRace();
  const cards = buildCards(chosen, results);
  state.animating = true;
  stopBtn.disabled = false;

  // Lockstep replay: each frame reveals `perFrame` more expansions for every
  // still-running algorithm. The one with the shortest `visited` list reaches
  // its goal first — you literally watch the smarter search finish sooner.
  let frame = 0;
  const finishOrder = [];

  function step() {
    const perFrame = Number(speedEl.value);
    frame += perFrame;

    let allDone = true;
    for (const id in cards) {
      const c = cards[id];
      if (c.done) continue;

      const visited = c.trace.visited;
      const upto = Math.min(frame, visited.length);
      for (let i = c.visited.size; i < upto; i++) {
        c.visited.add(nodeKey(visited[i]));
      }
      c.card.querySelector(".exp").textContent = c.visited.size;

      if (upto >= visited.length) {
        c.done = true;
        finishOrder.push(c);
        finalizeCard(c, finishOrder.length);
      } else {
        allDone = false;
        drawCard(c, false);
      }
    }

    if (allDone) {
      state.animating = false;
      state.rafId = null;
      raceBtn.disabled = false;
      stopBtn.disabled = true;
      announceWinner(cards);
      return;
    }
    state.rafId = requestAnimationFrame(step);
  }
  statusEl.textContent = "Racing…";
  state.rafId = requestAnimationFrame(step);
}

function finalizeCard(c, finishRank) {
  drawCard(c, true);
  const t = c.trace;
  const cost = state.view === "map" ? `${Math.round(t.path_cost)} m` : t.path_cost;
  c.card.querySelector(".len").textContent = t.found ? t.path_length : "no path";
  c.card.querySelector(".cost").textContent = t.found ? cost : "—";
  c.card.querySelector(".time").textContent = `${t.time_ms} ms`;
  const badge = c.card.querySelector(".badge");
  badge.textContent = t.found ? `done #${finishRank}` : "no path found";
}

function announceWinner(cards) {
  // "Winner" = found a path with the fewest nodes expanded (least work).
  let best = null;
  for (const id in cards) {
    const c = cards[id];
    if (!c.trace.found) continue;
    if (!best || c.trace.nodes_expanded < best.trace.nodes_expanded) best = c;
  }
  if (best) {
    best.card.classList.add("winner");
    best.card.querySelector(".badge").textContent = "★ least work";
    const unit = state.view === "map" ? "intersections" : "cells";
    statusEl.innerHTML = `<b>${best.algo.label}</b> reached the goal expanding only <b>${best.trace.nodes_expanded}</b> ${unit}.`;
  } else {
    statusEl.textContent = state.view === "map"
      ? "No route found — the two points may be on disconnected roads. Try points closer together."
      : "No algorithm could reach the goal — clear a route through the walls.";
  }
}

// ================================================================ Pac-Man (deep RL)
// A real game: the agent learns to play from the screen via a CNN on the GPU.
// We render the board, animate recorded games, and poll training progress so you
// watch the agent improve checkpoint by checkpoint.

const PAC = { cell: 38, layout: null, agents: [], agent: "double", game: "pacman",
              params: {}, job: null, poll: null, play: null, lastPlayed: -1, ckMeta: [] };
const GHOST_COLORS = ["#ff4d5e", "#ff9ad5", "#56e0e0", "#ffb14e"];

const pacCanvas = () => document.getElementById("pacCanvas");
const gameStatusEl = () => document.getElementById("gameStatus");

let gameLoaded = false;
async function initGame() {
  if (!PAC.layout) await loadLayout(PAC.game);
  if (!gameLoaded) { await loadGameAgents(); gameLoaded = true; }
  refreshModels();
}

async function loadLayout(game) {
  PAC.layout = await (await api(`/api/game_layout?game=${game}`)).json();
  sizeCanvas();
  drawBoard(null);
}

function sizeCanvas() {
  // keep the board a comfortable size for either game's aspect ratio
  PAC.cell = Math.max(16, Math.floor(420 / PAC.layout.cols));
  const c = pacCanvas();
  c.width = PAC.layout.cols * PAC.cell;
  c.height = PAC.layout.rows * PAC.cell;
}

document.getElementById("gameSelect").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.game === PAC.game) return;
  selectGame(btn.dataset.game);
});

async function selectGame(game) {
  PAC.game = game;
  document.querySelectorAll("#gameSelect button").forEach((b) =>
    b.classList.toggle("active", b.dataset.game === game));
  // reset playback / scrubber / status when switching games
  if (PAC.play) { cancelAnimationFrame(PAC.play.raf); PAC.play = null; }
  if (PAC.poll) { clearInterval(PAC.poll); PAC.poll = null; }
  document.querySelector(".ckscrub").hidden = true;
  document.getElementById("playBtn").disabled = true;
  document.getElementById("gameScore").textContent = "Train an agent to watch it play.";
  document.getElementById("gameStatus").textContent = "";
  document.getElementById("gameDevice").textContent = "";
  document.getElementById("gameCurve").getContext("2d").clearRect(0, 0, 999, 999);
  const nice = game === "pong" ? "Pong" : "Pac-Man";
  document.getElementById("gameTitle").innerHTML =
    `${nice} <span class="hint">(the agent learns to play from the screen — no rules given)</span>`;
  await loadLayout(game);
}

async function loadGameAgents() {
  PAC.agents = (await (await api("/api/game_agents")).json()).agents;
  const list = document.getElementById("gameAgentList");
  list.innerHTML = "";
  for (const ag of PAC.agents) {
    const item = document.createElement("div");
    item.className = "algo-item" + (ag.id === PAC.agent ? " selected" : "");
    item.dataset.id = ag.id;
    item.innerHTML = `
      <div class="row">
        <span class="swatch" style="background:var(--warn)"></span>
        <b>${ag.label}</b><span class="fam">${ag.family}</span>
      </div>
      <p>${ag.blurb}</p>
      <div class="item-actions">
        <button class="codebtn" data-id="${ag.id}">&lt;/&gt; see the code</button>
      </div>
      <details class="how" onclick="event.stopPropagation()"><summary>ⓘ how it works</summary><div class="how-body">${ag.info}</div></details>`;
    item.onclick = () => selectGameAgent(ag.id);
    item.querySelector(".codebtn").onclick = (e) => { e.stopPropagation(); openCode(ag.id, ag.label); };
    list.appendChild(item);
  }
  selectGameAgent(PAC.agent);
}

function selectGameAgent(id) {
  PAC.agent = id;
  document.querySelectorAll("#gameAgentList .algo-item").forEach((el) =>
    el.classList.toggle("selected", el.dataset.id === id));
  const ag = PAC.agents.find((a) => a.id === id);
  const box = document.getElementById("hyperBox");
  box.innerHTML = "<div class='hyper-title'>Hyperparameters</div>";
  PAC.params = {};
  for (const hp of ag.hyperparams) {
    PAC.params[hp.name] = hp.default;
    const row = document.createElement("label");
    row.className = "hyper-row";
    row.title = hp.help;
    row.innerHTML = `<span class="hp-name">${hp.label}</span>
      <input type="range" min="${hp.min}" max="${hp.max}" step="${hp.step}" value="${hp.default}" />
      <span class="hp-val">${hp.default}</span>`;
    const input = row.querySelector("input"), val = row.querySelector(".hp-val");
    input.oninput = () => {
      const v = Number(input.value);
      PAC.params[hp.name] = v;
      val.textContent = v;
    };
    box.appendChild(row);
  }
}

// --- rendering -------------------------------------------------------------
function drawBoard(frame) {
  if (PAC.layout && PAC.layout.game === "pong") return drawPong(frame);
  const c = pacCanvas(), ctx = c.getContext("2d"), cell = PAC.cell;
  ctx.fillStyle = "#05060a";
  ctx.fillRect(0, 0, c.width, c.height);
  // walls
  ctx.fillStyle = "#1d3a8f";
  for (const [r, col] of PAC.layout.walls) {
    ctx.fillRect(col * cell + 2, r * cell + 2, cell - 4, cell - 4);
  }
  if (!frame) return;
  // pellets
  ctx.fillStyle = "#ffd9a0";
  for (const [r, col] of frame.pellets) {
    ctx.beginPath(); ctx.arc(col * cell + cell / 2, r * cell + cell / 2, cell * 0.08, 0, 7); ctx.fill();
  }
  // power pellets
  for (const [r, col] of frame.power) {
    ctx.beginPath(); ctx.arc(col * cell + cell / 2, r * cell + cell / 2, cell * 0.2, 0, 7); ctx.fill();
  }
  // ghosts
  frame.ghosts.forEach((g, i) => drawGhost(ctx, cell, g.pos, g.scared ? "#3aa0ff" : GHOST_COLORS[i % 4], g.scared));
  // pac-man
  drawPacman(ctx, cell, frame.pac);
}

function drawPacman(ctx, cell, [r, c]) {
  const x = c * cell + cell / 2, y = r * cell + cell / 2, rad = cell * 0.42;
  ctx.fillStyle = "#ffe14d";
  ctx.beginPath();
  ctx.arc(x, y, rad, 0.25 * Math.PI, 1.75 * Math.PI);
  ctx.lineTo(x, y);
  ctx.fill();
}

function drawGhost(ctx, cell, [r, c], color, scared) {
  const x = c * cell + cell / 2, y = r * cell + cell / 2, rad = cell * 0.4;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y - rad * 0.15, rad, Math.PI, 0);
  ctx.lineTo(x + rad, y + rad * 0.7);
  for (let k = 0; k < 3; k++) ctx.lineTo(x + rad - (k * 2 + 1) * rad / 3, y + rad * (k % 2 ? 0.7 : 0.45));
  ctx.lineTo(x - rad, y + rad * 0.7);
  ctx.closePath(); ctx.fill();
  // eyes
  ctx.fillStyle = "#fff";
  for (const ex of [-rad * 0.4, rad * 0.4]) {
    ctx.beginPath(); ctx.arc(x + ex, y - rad * 0.1, rad * 0.22, 0, 7); ctx.fill();
  }
  ctx.fillStyle = scared ? "#fff" : "#12203f";
  for (const ex of [-rad * 0.4, rad * 0.4]) {
    ctx.beginPath(); ctx.arc(x + ex, y - rad * 0.1, rad * 0.1, 0, 7); ctx.fill();
  }
}

function drawPong(frame) {
  const c = pacCanvas(), ctx = c.getContext("2d"), cell = PAC.cell;
  const lay = PAC.layout;
  ctx.fillStyle = "#05060a";
  ctx.fillRect(0, 0, c.width, c.height);
  // centre net
  ctx.strokeStyle = "#2a3350"; ctx.lineWidth = 2; ctx.setLineDash([6, 8]);
  ctx.beginPath(); ctx.moveTo(c.width / 2, 0); ctx.lineTo(c.width / 2, c.height); ctx.stroke();
  ctx.setLineDash([]);
  if (!frame) return;
  const ph = lay.ph;
  // agent paddle (left, cyan) and opponent paddle (right, red)
  ctx.fillStyle = "#56e0e0";
  ctx.fillRect(frame.agent_col * cell + cell * 0.2, frame.agent_y * cell, cell * 0.6, ph * cell);
  ctx.fillStyle = "#ff5c7a";
  ctx.fillRect(frame.opp_col * cell + cell * 0.2, frame.opp_y * cell, cell * 0.6, ph * cell);
  // ball
  ctx.fillStyle = "#ffe14d";
  ctx.beginPath();
  ctx.arc(frame.ball[1] * cell + cell / 2, frame.ball[0] * cell + cell / 2, cell * 0.32, 0, 7);
  ctx.fill();
  // scoreline
  ctx.fillStyle = "#e6e9f0"; ctx.font = `bold ${Math.floor(cell * 0.9)}px system-ui`;
  ctx.textAlign = "center";
  ctx.fillText(frame.score[0], c.width * 0.35, cell * 1.1);
  ctx.fillText(frame.score[1], c.width * 0.65, cell * 1.1);
}

function drawGameCurve(scores) {
  const ctx = document.getElementById("gameCurve").getContext("2d");
  const w = ctx.canvas.width, h = ctx.canvas.height;
  ctx.fillStyle = T("--cell", "#11141b"); ctx.fillRect(0, 0, w, h);
  if (scores.length < 2) return;
  const lo = Math.min(...scores), hi = Math.max(...scores, lo + 1);
  ctx.strokeStyle = T("--curve-score", "#ffe14d"); ctx.lineWidth = 1.5; ctx.beginPath();
  scores.forEach((s, i) => {
    const px = (i / (scores.length - 1)) * (w - 6) + 3;
    const py = h - 4 - ((s - lo) / (hi - lo)) * (h - 8);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
}

// --- playback --------------------------------------------------------------
function playFrames(frames, fps = 11) {
  if (PAC.play) cancelAnimationFrame(PAC.play.raf);
  const playBtn = document.getElementById("playBtn");
  playBtn.disabled = false;
  const interval = 1000 / fps;
  const st = { frames, i: 0, last: 0, raf: null };
  PAC.play = st;
  function tick(t) {
    if (!st.last) st.last = t;
    if (t - st.last >= interval) {
      st.last = t;
      const f = st.frames[st.i];
      drawBoard(f);
      const scoreTxt = Array.isArray(f.score) ? `${f.score[0]}–${f.score[1]}` : `score ${f.score}`;
      document.getElementById("gameScore").textContent =
        `${scoreTxt}   ·   step ${f.step}` + (f.done && f.outcome ? `   ·   ${f.outcome.toUpperCase()}` : "");
      st.i++;
      if (st.i >= st.frames.length) return;   // stop at the end
    }
    st.raf = requestAnimationFrame(tick);
  }
  st.raf = requestAnimationFrame(tick);
}

document.getElementById("playBtn").onclick = () => {
  if (PAC.play) playFrames(PAC.play.frames);
};

async function fetchAndPlay(index) {
  const ck = await (await api(`/api/game_checkpoint?job_id=${PAC.job}&index=${index}`)).json();
  if (ck.frames) playFrames(ck.frames);
  return ck;
}

// --- training --------------------------------------------------------------
document.getElementById("gameTrainBtn").onclick = async () => {
  const btn = document.getElementById("gameTrainBtn");
  btn.disabled = true;
  gameStatusEl().textContent = "Starting training on the GPU…";
  document.querySelector(".ckscrub").hidden = true;
  PAC.lastPlayed = -1;

  let data;
  try {
    const res = await api("/api/game_train", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game: PAC.game, agent: PAC.agent, params: PAC.params }),
    });
    data = await res.json();
    if (!res.ok || data.error) {
      gameStatusEl().textContent = data.error || `Server refused (HTTP ${res.status}).`;
      btn.disabled = false; return;
    }
  } catch (e) {
    gameStatusEl().innerHTML = "⚠ Couldn't reach the training server — it may have restarted. Reload the page (Ctrl/Cmd-R) and try again.";
    btn.disabled = false; return;
  }

  PAC.job = data.job_id;
  PAC.layout = data.layout;
  pollFails = 0;
  sizeCanvas();
  drawBoard(null);

  if (PAC.poll) clearInterval(PAC.poll);
  PAC.poll = setInterval(() => pollGame(btn), 1000);
};

let polling = false;
let pollFails = 0;
async function pollGame(btn) {
  if (polling) return;
  polling = true;
  try {
    const res = await api(`/api/game_progress?job_id=${PAC.job}`);
    if (!res.ok) throw new Error("progress " + res.status);
    const p = await res.json();
    pollFails = 0;
    drawGameCurve(p.curve);
    PAC.ckMeta = p.checkpoints;
    const dev = p.device ? ` · ${p.device}` : "";
    gameStatusEl().innerHTML = `Episode <b>${p.episode}/${p.episodes}</b> · ${p.curve.length} games played${dev}`;

    // watch the most recent recorded game as it improves
    if (p.latest_index > PAC.lastPlayed && (!PAC.play || PAC.play.i >= PAC.play.frames.length)) {
      PAC.lastPlayed = p.latest_index;
      fetchAndPlay(p.latest_index);
    }

    if (p.status === "done" || p.status === "error") {
      clearInterval(PAC.poll); PAC.poll = null;
      btn.disabled = false;
      if (p.status === "error") { gameStatusEl().textContent = "Training error: " + p.error; return; }
      finishGame(p);
    }
  } catch (e) {
    // server may have dropped mid-training; stop after a few failures
    if (++pollFails >= 4) {
      clearInterval(PAC.poll); PAC.poll = null; btn.disabled = false;
      gameStatusEl().innerHTML = "⚠ Lost contact with the server during training (it may have crashed/restarted). Reload the page and try again — saved models are safe on disk.";
    }
  } finally { polling = false; }
}

function finishGame(p) {
  document.getElementById("gameDevice").textContent =
    `Trained ${p.episodes} games on ${p.device}.`;
  // best checkpoint by score
  let bestIdx = 0, bestScore = -1e9;
  p.checkpoints.forEach((c, i) => { if (c.score > bestScore) { bestScore = c.score; bestIdx = i; } });
  gameStatusEl().innerHTML = `Done. Best game scored <b>${bestScore}</b> (${p.checkpoints[bestIdx].outcome}). Drag the slider to watch any stage.`;

  // set up the checkpoint scrubber
  const scrub = document.getElementById("ckScrub"), wrap = document.querySelector(".ckscrub");
  wrap.hidden = false;
  scrub.max = p.checkpoints.length - 1;
  scrub.value = bestIdx;
  const setLabel = (i) => {
    const c = p.checkpoints[i];
    document.getElementById("ckLabel").textContent = `episode ${c.episode} — score ${c.score} (${c.outcome})`;
  };
  setLabel(bestIdx);
  scrub.oninput = () => { setLabel(Number(scrub.value)); fetchAndPlay(Number(scrub.value)); };
  fetchAndPlay(bestIdx);

  // offer to save the trained model
  const bar = document.getElementById("saveBar");
  bar.hidden = false;
  document.getElementById("modelName").value =
    `${PAC.game}-${PAC.agent}-${p.episodes}ep`;
}

// --- code viewer -----------------------------------------------------------
async function openCode(id, label) {
  const modal = document.getElementById("codeModal");
  document.getElementById("codeTitle").textContent = `${label} — source`;
  document.getElementById("codeContent").textContent = "loading…";
  modal.hidden = false;
  const data = await (await api(`/api/agent_code?id=${id}`)).json();
  document.getElementById("codeContent").textContent = data.code || data.error || "";
}
document.getElementById("codeClose").onclick = () => { document.getElementById("codeModal").hidden = true; };
document.getElementById("codeModal").onclick = (e) => {
  if (e.target.id === "codeModal") e.target.hidden = true;   // click backdrop to close
};

// --- saving / loading models ----------------------------------------------
document.getElementById("saveModelBtn").onclick = async () => {
  if (!PAC.job) return;
  const name = document.getElementById("modelName").value || "model";
  const r = await (await api("/api/model_save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: PAC.job, name }),
  })).json();
  if (r.ok) { document.getElementById("gameStatus").innerHTML += ` · saved as <b>${r.name}</b>`; refreshModels(); }
  else { gameStatusEl().textContent = r.error || "save failed"; }
};

async function refreshModels() {
  const data = await (await api("/api/models")).json();
  const box = document.getElementById("savedModels");
  if (!data.models.length) { box.innerHTML = `<span class="muted tiny-note">none yet — train one and save it</span>`; return; }
  box.innerHTML = "";
  for (const m of data.models) {
    const row = document.createElement("div");
    row.className = "saved-row";
    const score = m.best_score == null ? "" : ` · best ${m.best_score}`;
    row.innerHTML = `
      <div class="saved-meta"><b>${m.name}</b><span class="muted tiny-note">${m.game} · ${m.variant} · ${m.episodes}ep${score}</span></div>
      <div class="saved-btns">
        <button class="action testbtn">${icon("play")} Watch</button>
        <button class="action delbtn" aria-label="Delete this model" title="Delete">${icon("trash")}</button>
      </div>`;
    row.querySelector(".testbtn").onclick = () => testModel(m.name);
    row.querySelector(".delbtn").onclick = async () => {
      await api("/api/model_delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: m.name }) });
      refreshModels();
    };
    box.appendChild(row);
  }
}

async function testModel(name) {
  gameStatusEl().textContent = `Loading ${name}…`;
  const data = await (await api("/api/model_test", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  })).json();
  if (data.error) { gameStatusEl().textContent = data.error; return; }
  PAC.layout = data.layout;
  PAC.game = data.layout.game;
  sizeCanvas();
  playFrames(data.roll.frames);
  const sc = Array.isArray(data.roll.score) ? data.roll.score.join("–") : data.roll.score;
  gameStatusEl().innerHTML = `Replaying saved model <b>${name}</b> — scored ${sc} (${data.roll.outcome}).`;
}

// ================================================================ Teach an LLM
// Pipeline: load model → (teacher key) → SFT distill → DPO (RL) → chat test.
// Long ops are one-at-a-time background jobs polled via /api/llm/status.

const LLM = { polling: null };

async function llmStatus() { return (await api("/api/llm/status")).json(); }

async function initLlm() {
  const s = await llmStatus();
  renderLlmStatus(s);
  renderScorecard(s.eval);
  // reflect provider from env if any
  if (s.teacher) document.getElementById("llmProvider").value = s.teacher;
}

function renderLlmStatus(s) {
  document.getElementById("llmModelStatus").textContent =
    s.loaded ? `loaded on ${s.device}${s.has_tuned ? " · tuned ✓" : ""}` : "not loaded";
  document.getElementById("sftStatus").textContent =
    `${s.sft_examples} generated + built-in seed examples`;
  document.getElementById("dpoStatus").textContent =
    s.pref_pairs ? `${s.pref_pairs} preference pairs ready` : "needs preference pairs (teacher key)";
}

function llmDrawCurve(curve) {
  const ctx = document.getElementById("llmCurve").getContext("2d");
  const w = ctx.canvas.width, h = ctx.canvas.height;
  ctx.fillStyle = T("--cell", "#11141b"); ctx.fillRect(0, 0, w, h);
  if (!curve || curve.length < 2) return;
  const lo = Math.min(...curve), hi = Math.max(...curve, lo + 1e-6);
  ctx.strokeStyle = T("--curve-loss", "#3ad29f"); ctx.lineWidth = 1.5; ctx.beginPath();
  curve.forEach((v, i) => {
    const px = (i / (curve.length - 1)) * (w - 4) + 2;
    const py = h - 3 - ((v - lo) / (hi - lo)) * (h - 6);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
}

// Start a job (POST) then poll status until it finishes.
async function llmJob(endpoint, body, label) {
  const st = document.getElementById("llmStatus");
  st.textContent = `${label}…`;
  const r = await (await api(endpoint, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  })).json();
  if (r.ok === false) { st.textContent = "⚠ " + (r.error || "could not start"); return; }
  if (LLM.polling) clearInterval(LLM.polling);
  LLM.polling = setInterval(async () => {
    let s;
    try { s = await llmStatus(); } catch { return; }
    renderLlmStatus(s);
    llmDrawCurve(s.curve);
    renderScorecard(s.eval);
    const j = s.job;
    st.innerHTML = `<b>${j.phase || label}</b> — ${j.step}/${j.total || "…"} ${j.info || ""}`;
    if (j.status === "done" || j.status === "error" || j.status === "idle") {
      clearInterval(LLM.polling); LLM.polling = null;
      st.innerHTML = j.status === "error" ? "⚠ " + j.error
        : `✓ ${j.phase || label} done.`;
    }
  }, 1000);
}

// --- evaluation scorecard ---
function cmpRow(label, base, tuned, higherBetter) {
  if (base == null || tuned == null) return "";
  const tunedBetter = higherBetter ? tuned > base : tuned < base;
  const bcls = tunedBetter ? "" : "win", tcls = tunedBetter ? "win" : "";
  return `<div class="sc-row"><span class="sc-label">${label}</span>
    <span class="sc-val ${bcls}">${base}</span><span class="sc-vs">vs</span>
    <span class="sc-val ${tcls}">${tuned}</span></div>`;
}

function renderScorecard(rep) {
  const el = document.getElementById("evalScorecard");
  if (!rep) { el.hidden = true; return; }
  el.hidden = false;
  let html = `<div class="sc-head">Held-out evaluation · ${rep.n} questions ${rep.judge_used ? "· judge ON" : "· intrinsic only"}</div>`;
  html += `<div class="sc-cols"><span></span><span>base</span><span></span><span>tuned ✨</span></div>`;
  if (rep.win_rate) {
    const wr = rep.win_rate;
    html += `<div class="sc-winrate"><b>Win rate (judge, position-swapped):</b>
      tuned <b class="win">${wr.tuned}</b> · tie ${wr.tie} · base <b>${wr.base}</b>  →  <b class="win">${wr.tuned_pct}% tuned</b></div>`;
  }
  if (rep.rubric) html += cmpRow("Rubric 1–5 (judge)", rep.rubric.base, rep.rubric.tuned, true);
  html += cmpRow("Perplexity (lower=better)", rep.perplexity.base, rep.perplexity.tuned, false);
  html += cmpRow("ROUGE-L vs reference (higher=better)", rep.rouge_l.base, rep.rouge_l.tuned, true);

  // every held-out question with both answers
  if (rep.samples && rep.samples.length) {
    const esc = (s) => (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    const items = rep.samples.map((s, i) => {
      let verdict = "";
      if (s.winner) {
        const cls = s.winner === "tuned" ? "win" : "";
        const rub = (s.rubric_base != null) ? ` · rubric ${s.rubric_base}→${s.rubric_tuned}` : "";
        verdict = `<span class="qa-verdict ${cls}">judge: ${s.winner}${rub}</span>`;
      }
      return `<div class="qa-item">
        <div class="qa-q">${i + 1}. ${esc(s.question)} ${verdict}</div>
        <div class="qa-ans"><div class="qa-col"><span class="qa-tag">base</span>${esc(s.base)}</div>
        <div class="qa-col tuned"><span class="qa-tag">tuned ✨</span>${esc(s.tuned)}</div></div>
      </div>`;
    }).join("");
    html += `<details class="qa-all"><summary>Show all ${rep.samples.length} evaluation Q&amp;A</summary>${items}</details>`;
  }
  el.innerHTML = html;
}

document.getElementById("evalBtn").onclick = () => llmJob("/api/llm/evaluate", { use_judge: false }, "evaluating (intrinsic)");
document.getElementById("evalJudgeBtn").onclick = () => llmJob("/api/llm/evaluate", { use_judge: true }, "evaluating + judging");

document.getElementById("llmLoadBtn").onclick = () => llmJob("/api/llm/load", {}, "loading model");
document.getElementById("genSftBtn").onclick = () =>
  llmJob("/api/llm/gen_sft", { n_per_topic: 8 }, "teacher writing data");
document.getElementById("trainSftBtn").onclick = () =>
  llmJob("/api/llm/train_sft", { epochs: 3, lr: 1e-4 }, "SFT training");
document.getElementById("genPrefsBtn").onclick = () =>
  llmJob("/api/llm/gen_prefs", { n: 24 }, "teacher judging pairs");
document.getElementById("trainDpoBtn").onclick = () =>
  llmJob("/api/llm/train_dpo", { epochs: 1, lr: 5e-5, beta: 0.1 }, "DPO training");

document.getElementById("llmKeyBtn").onclick = async () => {
  const status = document.getElementById("llmKeyStatus");
  status.textContent = "checking…";
  const r = await (await api("/api/llm/set_key", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider: document.getElementById("llmProvider").value,
                           api_key: document.getElementById("llmKey").value }),
  })).json();
  status.textContent = r.ok ? "✓ key works" : "⚠ " + (r.error || "invalid key");
};

document.getElementById("askBtn").onclick = async () => {
  const q = document.getElementById("llmQuestion").value.trim();
  if (!q) return;
  const base = document.getElementById("answerBase"), tuned = document.getElementById("answerTuned");
  base.textContent = "thinking…"; tuned.textContent = "thinking…";
  base.classList.add("muted"); tuned.classList.add("muted");
  const r = await (await api("/api/llm/chat", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q }),
  })).json();
  if (r.error) { base.textContent = "⚠ " + r.error; tuned.textContent = ""; return; }
  base.textContent = r.base; base.classList.remove("muted");
  tuned.textContent = r.tuned || "(no tuned model yet — train SFT first)";
  if (r.tuned) tuned.classList.remove("muted");
};

// ================================================================ Reasoning Lab
// A drag-and-drop graph of prompted agents (generator / critic / refiner …),
// each backed by the local model or the teacher API, run as a reasoning loop.

const REASON = { editor: null, types: {}, poll: null };

function rPorts(spec) {
  // inputs, outputs for a node type
  if (spec.kind === "io" && spec.ports.includes("out") && !spec.ports.includes("in")) return [0, 1]; // input
  if (spec.kind === "io") return [1, 0];        // output
  if (spec.kind === "critic") return [1, 2];     // pass / fail
  return [1, 1];                                  // gen / refine / consistency
}

function rNodeHtml(type, spec) {
  const critic = spec.kind === "critic";
  const io = spec.kind === "io";
  let body = `<div class="rn-title">${spec.label}</div>`;
  if (!io) {
    body += `<select df-model class="rn-model"><option value="local">local Qwen</option><option value="api">teacher API</option></select>`;
    body += `<textarea df-prompt class="rn-prompt" rows="3">${spec.prompt || ""}</textarea>`;
  }
  if (critic) {
    body += `<label class="rn-th">pass if score ≥ <input type="number" df-threshold min="1" max="5" value="4"></label>`;
    body += `<div class="rn-ports"><span class="ok">✓ pass</span><span class="no">✗ fail</span></div>`;
  }
  return `<div class="rn type-${spec.kind}">${body}</div>`;
}

function rAddNode(type, x = 60, y = 60) {
  const spec = REASON.types[type];
  const [ins, outs] = rPorts(spec);
  const data = spec.kind === "io" ? {} : { prompt: spec.prompt || "", model: "local", threshold: 4 };
  return REASON.editor.addNode(type, ins, outs, x, y, type, data, rNodeHtml(type, spec));
}

async function initReason() {
  if (!REASON.types || !Object.keys(REASON.types).length) {
    const d = await (await api("/api/reason/node_types")).json();
    d.types.forEach((t) => (REASON.types[t.id] = t));
    const pal = document.getElementById("reasonPalette");
    pal.innerHTML = "";
    for (const t of d.types) {
      const b = document.createElement("button");
      b.className = "action pal-btn"; b.textContent = "+ " + t.label;
      b.onclick = () => rAddNode(t.id, 40 + Math.random() * 120, 40 + Math.random() * 200);
      pal.appendChild(b);
    }
  }
  if (!REASON.editor) {
    REASON.editor = new Drawflow(document.getElementById("drawflow"));
    REASON.editor.reroute = true;
    REASON.editor.start();
    buildExampleGraph();
  } else {
    setTimeout(() => { try { REASON.editor.zoom_reset(); } catch (e) {} }, 50);
  }
}

function buildExampleGraph() {
  const e = REASON.editor;
  e.clear();
  const q = rAddNode("input", 30, 120);
  const g = rAddNode("generator", 270, 60);
  const c = rAddNode("relevance_critic", 560, 60);
  const r = rAddNode("refiner", 270, 300);
  const o = rAddNode("output", 850, 90);
  e.addConnection(q, g, "output_1", "input_1");
  e.addConnection(g, c, "output_1", "input_1");
  e.addConnection(c, o, "output_1", "input_1");   // pass → output
  e.addConnection(c, r, "output_2", "input_1");   // fail → refiner
  e.addConnection(r, c, "output_1", "input_1");   // loop back to critic
}

function buildCleanGraph() {
  const ex = REASON.editor.export().drawflow.Home.data;
  const nodes = [], links = [];
  for (const id in ex) {
    const n = ex[id], d = n.data || {};
    const type = n.class;
    const spec = REASON.types[type] || {};
    nodes.push({ id: String(id), type, prompt: d.prompt || "", model: d.model || "local",
                 threshold: Number(d.threshold) || 4 });
    const critic = spec.kind === "critic";
    for (const outKey in n.outputs) {
      const idx = parseInt(outKey.split("_")[1], 10);
      const port = critic ? (idx === 1 ? "pass" : "fail") : "out";
      for (const conn of n.outputs[outKey].connections) {
        links.push({ from: String(id), port, to: String(conn.node) });
      }
    }
  }
  return { nodes, links };
}

document.getElementById("reasonExampleBtn").onclick = () => buildExampleGraph();
document.getElementById("reasonClearBtn").onclick = () => REASON.editor.clear();

// guided "video" walkthrough: builds the graph step by step with captions
let demoRunning = false;
document.getElementById("reasonDemoBtn").onclick = async () => {
  if (demoRunning) return;
  demoRunning = true;
  const cap = document.getElementById("reasonCaption");
  const show = (t) => { cap.hidden = false; cap.innerHTML = t; };
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const e = REASON.editor;
  e.clear();
  try {
    show("① Click a behaviour to add an agent. We'll start with <b>Question in</b>.");
    await wait(1700);
    const q = rAddNode("input", 20, 150); await wait(1100);
    show("② Add a <b>Generator</b> — it writes the first answer.");
    await wait(1500);
    const g = rAddNode("generator", 250, 70); await wait(1100);
    show("③ Add a <b>Relevance critic</b> — it scores how on-topic the answer is (1–5).");
    await wait(1700);
    const c = rAddNode("relevance_critic", 530, 60); await wait(1100);
    show("④ Add a <b>Refiner</b> (rewrites using the critique) and an <b>Answer out</b>.");
    await wait(1600);
    const r = rAddNode("refiner", 250, 320); await wait(700);
    const o = rAddNode("output", 820, 110); await wait(1100);
    show("⑤ Wire it up: outputs → inputs. Critic <span class='ok'>✓ pass</span> → Answer, <span class='no'>✗ fail</span> → Refiner → back to the critic (the loop).");
    await wait(900);
    e.addConnection(q, g, "output_1", "input_1"); await wait(550);
    e.addConnection(g, c, "output_1", "input_1"); await wait(550);
    e.addConnection(c, o, "output_1", "input_1"); await wait(550);
    e.addConnection(c, r, "output_2", "input_1"); await wait(550);
    e.addConnection(r, c, "output_1", "input_1"); await wait(1600);
    show("⑥ Type a question and press <b>▶ Run graph</b>. The trace appears on the right →");
    document.getElementById("reasonQ").value = "When should I use PPO instead of DQN?";
    await wait(2200);
    document.getElementById("reasonRunBtn").click();
    await wait(2600);
    show("That's it! Edit any prompt, swap a node's model, or add more critics. ✨");
    await wait(3500);
    cap.hidden = true;
  } finally {
    demoRunning = false;
  }
};

document.getElementById("reasonRunBtn").onclick = async () => {
  const q = document.getElementById("reasonQ").value.trim();
  if (!q) return;
  const trace = document.getElementById("reasonTrace");
  document.getElementById("reasonFinal").hidden = true;
  trace.innerHTML = `<span class="muted tiny-note">running…</span>`;
  const r = await (await api("/api/reason/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph: buildCleanGraph(), question: q }),
  })).json();
  if (r.error) { trace.innerHTML = "⚠ " + r.error; return; }
  if (REASON.poll) clearInterval(REASON.poll);
  REASON.poll = setInterval(pollReason, 800);
};

function lightReasonNodes(steps) {
  document.querySelectorAll("#drawflow .drawflow-node").forEach((n) =>
    n.classList.remove("rn-lit", "rn-pass", "rn-fail"));
  if (!steps || !steps.length) return;
  const last = steps[steps.length - 1];
  const el = document.getElementById("node-" + last.id);
  if (!el) return;
  el.classList.add("rn-lit");
  if ((last.type && last.type.endsWith("critic")) || last.type === "verifier")
    el.classList.add(last.passed ? "rn-pass" : "rn-fail");
}

async function pollReason() {
  let s;
  try { s = await (await api("/api/reason/status")).json(); } catch { return; }
  renderTrace(s.steps);
  lightReasonNodes(s.steps);
  if (s.status === "done" || s.status === "error") {
    clearInterval(REASON.poll); REASON.poll = null;
    const fin = document.getElementById("reasonFinal");
    if (s.error) { document.getElementById("reasonTrace").innerHTML += `<div class="rt-step">⚠ ${s.error}</div>`; }
    else { fin.hidden = false; fin.innerHTML = `<div class="rf-head">Final answer · ${s.iterations} refine loop(s)</div>${escapeHtml(s.final)}`; }
  }
}

function escapeHtml(s) { return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

function renderTrace(steps) {
  const el = document.getElementById("reasonTrace");
  if (!steps || !steps.length) { el.innerHTML = `<span class="muted tiny-note">running…</span>`; return; }
  el.innerHTML = steps.map((s) => {
    let badge = "";
    if (s.type && s.type.endsWith("critic") || s.type === "verifier") {
      const cls = s.passed ? "ok" : "no";
      badge = `<span class="rt-badge ${cls}">${s.score}/${s.threshold} ${s.passed ? "✓ pass" : "✗ fail"}</span>`;
    } else if (s.model) {
      badge = `<span class="rt-badge">${s.model}</span>`;
    }
    return `<div class="rt-step"><div class="rt-head">${s.label} ${badge}</div><div class="rt-text">${escapeHtml(s.text || "")}</div></div>`;
  }).join("");
}

// ================================================================ University
// A Dean designs a curriculum; the model is taught each subject to mastery.
// Each mastered subject is saved as a skill on that model (toggle on/off, retrain).

const AUTO = { poll: null, target: 0.7, rebuildPoll: null };

const NODE_LABEL = { input: "Question", generator: "Generator", refiner: "Refiner",
  relevance_critic: "Relevance critic", correctness_critic: "Correctness critic",
  verifier: "Verifier", self_consistency: "Self-consistency", output: "Answer" };

function orderNodes(graph) {
  const byId = {}; graph.nodes.forEach((n) => (byId[n.id] = n));
  const next = {}; graph.links.forEach((l) => { if (!next[l.from]) next[l.from] = l.to; });
  const order = [], seen = new Set();
  let cur = (graph.nodes.find((n) => n.type === "input") || graph.nodes[0] || {}).id;
  while (cur && byId[cur] && !seen.has(cur)) { seen.add(cur); order.push(byId[cur]); cur = next[cur]; }
  graph.nodes.forEach((n) => { if (!seen.has(n.id)) order.push(n); });
  return order;
}

function renderAutoGraph(active) {
  const el = document.getElementById("autoGraph");
  if (!active || !active.graph) { el.innerHTML = ""; return; }
  const order = orderNodes(active.graph);
  el.innerHTML = order.map((n, i) => {
    const lit = active.node === n.id ? " lit" : "";
    const arrow = i < order.length - 1 ? '<span class="ag-arrow">→</span>' : "";
    const mdl = n.model ? `<span class="ag-mdl">${n.model === "api" ? "teacher" : n.model === "local" ? "student" : ""}</span>` : "";
    return `<div class="ag-node type-${n.type}${lit}">${NODE_LABEL[n.type] || n.type}${mdl}</div>${arrow}`;
  }).join("");
}

// ===================== Campus: a walkable virtual world ======================
// World view: agents WALK along paths between buildings. When everyone arrives
// (solo mode), the camera zooms into the building's interior. Cohort mode stays
// in the world so you can see each student choose lecture vs the common area.
const ROOMS = {
  dean:    { emoji: "🏛️", name: "Dean's office", wall: "#1b2433", floor: "#2a2030", desc: "designs the curriculum" },
  library: { emoji: "📚", name: "Library", wall: "#14211c", floor: "#2b2418", desc: "study & coursework" },
  lecture: { emoji: "🧑‍🏫", name: "Lecture hall", wall: "#191f2c", floor: "#22262f", desc: "exams" },
  dorm:    { emoji: "🛏️", name: "Dorm", wall: "#1d1830", floor: "#2a2436", desc: "rest / re-study" },
  grad:    { emoji: "🎓", name: "Graduation hall", wall: "#21220f", floor: "#2c2d18", desc: "graduation" },
  common:  { emoji: "☕", name: "Common area", wall: "#241c14", floor: "#2c2418", desc: "students socialise" },
};
const ROOM_SPOTS = {
  dean:    { student: [260, 250], teacher: [540, 250] },
  library: { student: [250, 255], teacher: [560, 250] },
  lecture: { student: [300, 262], teacher: [560, 250] },
  dorm:    { student: [250, 250], teacher: [560, 252] },
  grad:    { student: [360, 250], teacher: [520, 252] },
  common:  { student: [300, 255], teacher: [540, 250] },
};
// Exterior building positions on the campus map (canvas 800x340)
const WORLD = {
  hub: [400, 196],
  buildings: {
    dean:    { x: 112, y: 78 },
    library: { x: 400, y: 64 },
    lecture: { x: 688, y: 78 },
    dorm:    { x: 112, y: 300 },
    common:  { x: 400, y: 312 },
    grad:    { x: 688, y: 300 },
  },
  trees: [[218, 160], [582, 160], [60, 196], [740, 196], [258, 268], [542, 268]],
};
const WALK_SPEED = 3.0;

const CAMPUS = {
  ctx: null, frame: 0, pin: null,
  mode: "world", room: null, walkTarget: null, zoom: 0,
  agents: { student: { x: 360, y: 200, waypoints: [], emoji: "🤖", label: "Student" },
            teacher: { x: 440, y: 200, waypoints: [], emoji: "🧑‍🏫", label: "Teacher" } },
  bubble: { who: null, text: "" },
  banner: "Waiting to enrol…", subjects: [], board: "", glowKey: null, glowT: 0, confetti: [],
  cohort: false, students: [],
};

function campusInit() {
  const cv = document.getElementById("campus");
  if (!cv || CAMPUS.ctx) return;
  CAMPUS.ctx = cv.getContext("2d");
  const tabs = document.getElementById("campusTabs");
  if (tabs) {
    tabs.innerHTML = `<button data-room="" class="cam-tab on">▶ Auto-follow</button>` +
      `<button data-room="world" class="cam-tab">🌍 Campus map</button>` +
      Object.entries(ROOMS).map(([k, r]) => `<button data-room="${k}" class="cam-tab">${r.emoji} ${r.name}</button>`).join("");
    tabs.querySelectorAll(".cam-tab").forEach((b) => b.onclick = () => {
      CAMPUS.pin = b.dataset.room || null;
      tabs.querySelectorAll(".cam-tab").forEach((x) => x.classList.toggle("on",
        (x.dataset.room || "") === (CAMPUS.pin || "")));
    });
  }
  const loop = () => { campusDraw(); requestAnimationFrame(loop); };
  loop();
}

function _doorOf(roomKey, slot, total) {
  const b = WORLD.buildings[roomKey];
  const below = b.y < 196;                     // top row → door on the south side
  const off = (slot - (Math.max(1, total) - 1) / 2) * 34;
  return [b.x + off, b.y + (below ? 60 : -56)];
}

function walkTo(ag, roomKey, slot, total) {
  // walk via the central plaza, then to the building's door
  const door = _doorOf(roomKey, slot, total);
  ag.waypoints = [[WORLD.hub[0], WORLD.hub[1]], door];
  ag.dest = roomKey;
}

function stepAgent(ag) {
  if (!ag.waypoints.length) return true;       // idle / arrived
  const [tx, ty] = ag.waypoints[0];
  const dx = tx - ag.x, dy = ty - ag.y;
  const d = Math.hypot(dx, dy);
  if (d <= WALK_SPEED) { ag.x = tx; ag.y = ty; ag.waypoints.shift(); }
  else { ag.x += (dx / d) * WALK_SPEED; ag.y += (dy / d) * WALK_SPEED; }
  return !ag.waypoints.length;
}

function goRoom(room, sBub, tBub) {
  if (sBub) CAMPUS.bubble = { who: "student", text: sBub };
  else if (tBub) CAMPUS.bubble = { who: "teacher", text: tBub };
  if (room === CAMPUS.room && CAMPUS.mode === "interior") return;   // already there
  if (room === CAMPUS.walkTarget) return;                            // already walking
  CAMPUS.walkTarget = room;
  CAMPUS.zoom = 0;
  if (CAMPUS.mode === "interior" && CAMPUS.room) {
    // step out of the old building first
    const out = _doorOf(CAMPUS.room, 0, 2), out2 = _doorOf(CAMPUS.room, 1, 2);
    CAMPUS.agents.student.x = out[0]; CAMPUS.agents.student.y = out[1];
    CAMPUS.agents.teacher.x = out2[0]; CAMPUS.agents.teacher.y = out2[1];
  }
  CAMPUS.mode = "world";
  walkTo(CAMPUS.agents.student, room, 0, 2);
  walkTo(CAMPUS.agents.teacher, room, 1, 2);
}

function campusScene(events, active) {
  if (!CAMPUS.ctx) return;
  const a = active || {};
  const evs = events || [];
  const last = evs.length ? evs[evs.length - 1] : null;
  const domain = (evs.find((e) => e.type === "start") || {}).domain || "the subject";
  const cur = [...evs].reverse().find((e) => e.type === "curriculum");
  CAMPUS.subjects = cur ? cur.subjects.map((s) => s.subject) : [];

  // ---- cohort mode: a competing class walks the campus ----
  const cohortStart = evs.find((e) => e.type === "cohort_start");
  if (cohortStart) {
    CAMPUS.cohort = true;
    if (CAMPUS.students.length !== cohortStart.students.length) {
      CAMPUS.students = cohortStart.students.map((s, i) => ({
        name: s.name, persona: s.persona, emoji: s.emoji,
        x: WORLD.hub[0] + (i - 1) * 50, y: WORLD.hub[1], waypoints: [], dest: null }));
    }
    const w = [...evs].reverse().find((e) => e.type === "cohort_where");
    // where should each student be right now?
    let teacherDest = "lecture";
    let dests = null;
    if (last.type === "cohort_start" || last.type === "curriculum") { dests = "dean"; teacherDest = "dean"; }
    else if (last.type === "cohort_leaderboard" || last.type === "final" || last.type === "thesis") {
      dests = "grad"; teacherDest = "grad";
      CAMPUS.glowKey = "grad"; CAMPUS.glowT = 150;
      if (!CAMPUS._conf && last.type !== "thesis") { _spawnConfetti(); CAMPUS._conf = true; }
    }
    CAMPUS.students.forEach((st, i) => {
      const want = dests || (w ? (w.where[st.name] === "common" ? "common" : "lecture") : "lecture");
      if (st.dest !== want) walkTo(st, want, i, CAMPUS.students.length);
    });
    if (CAMPUS.agents.teacher.dest !== teacherDest)
      walkTo(CAMPUS.agents.teacher, teacherDest, 1, 2);
    CAMPUS.board = a.question || CAMPUS.board;
    if (a.question) CAMPUS.banner = `📝 Exam · ${a.subject || ""}`;
    else if (last.type === "cohort_where") CAMPUS.banner = `📚 Lecture: ${last.subject} — who shows up?`;
    else if (last.type === "subject_start") CAMPUS.banner = `📖 ${last.subject}`;
    else if (last.type === "cohort_subject") CAMPUS.banner = `✍️ ${last.subject} graded`;
    else if (last.type === "cohort_leaderboard" || last.type === "final") CAMPUS.banner = `🎓 Results are in!`;
    else CAMPUS.banner = `🎓 ${cohortStart.n_students}-student cohort · ${cohortStart.domain || ""}`;
    renderCohortBoard(evs);
    return;
  }

  if (a.question) {
    CAMPUS.banner = `📝 Exam · ${a.subject || ""}`;
    CAMPUS.board = a.question;
    if (a.node === "c") goRoom("lecture", "", "Let me grade this…");
    else if (a.node === "r") goRoom("lecture", "Improving my answer…", "");
    else goRoom("lecture", "Answering…", "");
    return;
  }
  if (!last) { CAMPUS.banner = "Waiting to enrol…"; return; }
  switch (last.type) {
    case "start":
      CAMPUS.banner = `🚀 Enrolled in ${domain}`; goRoom("dean", "", "Welcome! Let's plan your studies."); break;
    case "curriculum":
      CAMPUS.banner = `🏛️ ${last.subjects.length}-subject curriculum`; goRoom("dean", "", `${last.subjects.length} subjects to master!`); break;
    case "replan":
      CAMPUS.banner = `🔁 Re-planning weak subjects`; goRoom("dean", "", "Let's revisit the hard parts."); break;
    case "subject_start":
      CAMPUS.banner = `📖 Next: ${last.subject}`; CAMPUS.board = last.subject; goRoom("library", `Off to study ${last.subject}…`, ""); break;
    case "study":
      CAMPUS.banner = `📚 Studying ${last.subject}`; CAMPUS.board = last.subject; goRoom("library", "Reading & training…", ""); break;
    case "exam":
      CAMPUS.banner = `✍️ ${last.subject}: ${(last.mastery * 100).toFixed(0)}%`; goRoom("lecture", "", `Score: ${(last.mastery * 100).toFixed(0)}%`); break;
    case "subject_done":
      if (last.mastered) {
        CAMPUS.banner = `🎓 Mastered ${last.subject}!`; goRoom("grad", `Mastered ${last.subject}! 🎉`, "");
        CAMPUS.glowKey = "grad"; CAMPUS.glowT = 120; _spawnConfetti();
      } else { CAMPUS.banner = `🛏️ ${last.subject}: needs more study`; goRoom("dorm", "Need more practice…", ""); }
      break;
    case "thesis":
      CAMPUS.banner = `📜 Writing graduation thesis…`; goRoom("library", "Writing my thesis… ✍️", ""); break;
    case "final":
      CAMPUS.banner = `🏁 Term complete · GPA ${((last.gpa || 0) * 100).toFixed(0)}%`; goRoom("grad", "Graduated! 🎓", "Well done.");
      CAMPUS.glowKey = "grad"; CAMPUS.glowT = 200; _spawnConfetti(); break;
    case "stopped":
      CAMPUS.banner = `■ Stopped (${last.reason})`; goRoom("dorm", "Resting…", ""); break;
    default:
      CAMPUS.banner = "…";
  }
}

function _spawnConfetti() {
  const cols = ["#f87171", "#fbbf24", "#34d399", "#60a5fa", "#c084fc"];
  for (let i = 0; i < 40; i++)
    CAMPUS.confetti.push({ x: 100 + Math.random() * 600, y: -Math.random() * 100,
      vy: 1.5 + Math.random() * 2, c: cols[i % cols.length], s: 3 + Math.random() * 3 });
}

function _roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function _shelf(ctx, x, y, w) {
  ctx.fillStyle = "#3a2c1c"; ctx.fillRect(x, y, w, 44);
  const cols = ["#7f1d1d", "#1e3a5f", "#14532d", "#713f12", "#4c1d95"];
  for (let i = 0; i < Math.floor(w / 9); i++) {
    ctx.fillStyle = cols[(i + Math.floor(x)) % cols.length];
    ctx.fillRect(x + 3 + i * 9, y + 4, 6, 18);
    ctx.fillRect(x + 3 + i * 9, y + 24, 6, 16);
  }
}

function _char(ctx, ag, label, off) {
  const bob = Math.sin((CAMPUS.frame + off) / 16) * 3;
  ctx.textAlign = "center"; ctx.font = "44px system-ui";
  ctx.fillText(ag.emoji, ag.x, ag.y + bob);
  ctx.fillStyle = "#9fb0c8"; ctx.font = "11px system-ui";
  ctx.fillText(label, ag.x, ag.y + 22);
}

function _bubbleOver(ctx, x, y, text, W, maxw) {
  ctx.font = "13px system-ui";
  const lines = _wrapText(ctx, text, maxw || 230);
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 18;
  const h = lines.length * 17 + 12;
  let bx = x - w / 2, by = y - 52 - h;
  bx = Math.max(6, Math.min(W - w - 6, bx)); by = Math.max(34, by);
  ctx.fillStyle = "#eef3fb"; ctx.strokeStyle = "#c7d2e0"; ctx.lineWidth = 1;
  _roundRect(ctx, bx, by, w, h, 9); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x - 6, by + h); ctx.lineTo(x + 6, by + h); ctx.lineTo(x, by + h + 9); ctx.fill();
  ctx.fillStyle = "#1a2330"; ctx.textAlign = "left";
  lines.forEach((l, i) => ctx.fillText(l, bx + 9, by + 18 + i * 17));
}

function _drawHouse(ctx, key) {
  const b = WORLD.buildings[key], r = ROOMS[key];
  const glow = CAMPUS.glowKey === key && CAMPUS.glowT > 0;
  // facade
  ctx.fillStyle = glow ? "#1d3b2a" : "#1d2733";
  ctx.strokeStyle = glow ? "#34d399" : "#33414f";
  ctx.lineWidth = glow ? 3 : 1.5;
  _roundRect(ctx, b.x - 52, b.y - 22, 104, 60, 6); ctx.fill(); ctx.stroke();
  // roof
  ctx.fillStyle = glow ? "#2a5a3f" : "#2b3a48";
  ctx.beginPath(); ctx.moveTo(b.x - 60, b.y - 20); ctx.lineTo(b.x, b.y - 52); ctx.lineTo(b.x + 60, b.y - 20); ctx.closePath(); ctx.fill();
  // door (faces the plaza)
  const below = b.y < 196;
  ctx.fillStyle = "#0c1118";
  ctx.fillRect(b.x - 11, below ? b.y + 16 : b.y - 22, 22, 22);
  // sign + label
  ctx.textAlign = "center"; ctx.font = "22px system-ui";
  ctx.fillText(r.emoji, b.x, b.y + 4);
  ctx.fillStyle = "#9fb0c8"; ctx.font = "11px system-ui";
  ctx.fillText(r.name, b.x, below ? b.y + 52 : b.y - 60);
}

function _drawWalker(ctx, ag, idx) {
  const walking = ag.waypoints && ag.waypoints.length;
  const bob = walking ? Math.abs(Math.sin((CAMPUS.frame + idx * 5) / 5)) * -4
                      : Math.sin((CAMPUS.frame + idx * 7) / 16) * 2;
  ctx.textAlign = "center"; ctx.font = "30px system-ui";
  ctx.fillText(ag.emoji || "🤖", ag.x, ag.y + bob);
  ctx.fillStyle = "#cdd6e6"; ctx.font = "10px system-ui";
  ctx.fillText(ag.label || (ag.name ? "#" + ag.name.split("#").pop() : ""), ag.x, ag.y + 14);
  if (ag.persona) { ctx.fillStyle = "#7c8aa0"; ctx.font = "9px system-ui"; ctx.fillText(ag.persona, ag.x, ag.y + 25); }
}

function drawWorld(ctx, cv) {
  const W = cv.width, H = cv.height;
  // grass + subtle texture
  ctx.fillStyle = "#10241a"; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "rgba(255,255,255,.02)";
  for (let i = 0; i < 24; i++) ctx.fillRect((i * 137) % W, (i * 89) % H, 26, 12);
  // paths from each door to the plaza
  ctx.strokeStyle = "#2e3a40"; ctx.lineWidth = 18; ctx.lineCap = "round";
  Object.keys(WORLD.buildings).forEach((k) => {
    const d = _doorOf(k, 0.5, 2);
    ctx.beginPath(); ctx.moveTo(WORLD.hub[0], WORLD.hub[1]); ctx.lineTo(d[0], d[1]); ctx.stroke();
  });
  // plaza + fountain
  ctx.fillStyle = "#2e3a40"; ctx.beginPath(); ctx.arc(WORLD.hub[0], WORLD.hub[1], 34, 0, Math.PI * 2); ctx.fill();
  ctx.textAlign = "center"; ctx.font = "20px system-ui"; ctx.fillText("⛲", WORLD.hub[0], WORLD.hub[1] + 7);
  // trees
  ctx.font = "20px system-ui";
  WORLD.trees.forEach(([x, y]) => ctx.fillText("🌳", x, y));
  // buildings
  Object.keys(WORLD.buildings).forEach((k) => _drawHouse(ctx, k));

  // agents: cohort students + teacher, or solo pair
  const walkers = CAMPUS.cohort
    ? [...CAMPUS.students, CAMPUS.agents.teacher]
    : [CAMPUS.agents.student, CAMPUS.agents.teacher];
  walkers.forEach((ag, i) => { stepAgent(ag); _drawWalker(ctx, ag, i); });

  // bubble over the speaker (matched by name for cohort)
  const bub = CAMPUS.bubble;
  if (bub.who && bub.text) {
    const ag = CAMPUS.cohort
      ? (CAMPUS.students.find((s) => s.name === bub.who) || (bub.who === "teacher" ? CAMPUS.agents.teacher : null))
      : CAMPUS.agents[bub.who];
    if (ag) _bubbleOver(ctx, ag.x, ag.y, bub.text, W, 200);
  }

  // banner + footer
  ctx.fillStyle = "#e8eef7"; ctx.font = "600 15px system-ui"; ctx.textAlign = "left";
  ctx.fillText(CAMPUS.banner || "", 14, 26);
  ctx.fillStyle = "#7c8aa0"; ctx.font = "11px system-ui";
  ctx.fillText(`🌍 Campus${CAMPUS.pin ? "  (pinned — Auto-follow off)" : ""}`, 14, H - 12);

  if (CAMPUS.confetti.length) {
    CAMPUS.confetti.forEach((p) => { p.y += p.vy; ctx.fillStyle = p.c; ctx.fillRect(p.x, p.y, p.s, p.s); });
    CAMPUS.confetti = CAMPUS.confetti.filter((p) => p.y < H);
  }
  if (CAMPUS.glowT > 0) CAMPUS.glowT--;
}

function drawInterior(ctx, cv, roomKey) {
  const W = cv.width, H = cv.height, floorY = 210;
  const room = ROOMS[roomKey] || ROOMS.dean;

  // walls + floor
  ctx.fillStyle = room.wall; ctx.fillRect(0, 0, W, floorY);
  ctx.fillStyle = room.floor; ctx.fillRect(0, floorY, W, H - floorY);
  ctx.strokeStyle = "rgba(255,255,255,.04)"; ctx.lineWidth = 2;
  for (let i = 1; i < 6; i++) { ctx.beginPath(); ctx.moveTo(i * W / 6, floorY); ctx.lineTo(i * W / 6 - 30, H); ctx.stroke(); }
  ctx.strokeStyle = "rgba(255,255,255,.07)"; ctx.beginPath(); ctx.moveTo(0, floorY); ctx.lineTo(W, floorY); ctx.stroke();

  // per-room furniture
  if (roomKey === "library") {
    _shelf(ctx, 30, 60, 200); _shelf(ctx, 250, 60, 200); _shelf(ctx, 560, 60, 210);
    ctx.fillStyle = "#3a2c1c"; ctx.fillRect(210, 250, 150, 36); ctx.fillRect(470, 250, 150, 36);
  } else if (roomKey === "lecture") {
    ctx.fillStyle = "#14361f"; _roundRect(ctx, 120, 40, 560, 120, 8); ctx.fill();
    ctx.strokeStyle = "#5b3a1a"; ctx.lineWidth = 6; ctx.stroke();
    ctx.fillStyle = "#d7e6da"; ctx.font = "600 15px system-ui"; ctx.textAlign = "center";
    _wrapText(ctx, "Q: " + (CAMPUS.board || "…"), 520).slice(0, 3).forEach((l, i) => ctx.fillText(l, 400, 80 + i * 24));
  } else if (roomKey === "dean") {
    ctx.fillStyle = "#0f1622"; _roundRect(ctx, 470, 40, 300, 150, 8); ctx.fill();
    ctx.strokeStyle = "#33415a"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#cdd6e6"; ctx.font = "600 13px system-ui"; ctx.textAlign = "left";
    ctx.fillText("📋 Curriculum", 486, 62);
    ctx.font = "11px system-ui"; ctx.fillStyle = "#9fb0c8";
    CAMPUS.subjects.slice(0, 7).forEach((s, i) => ctx.fillText("• " + s.slice(0, 34), 486, 82 + i * 15));
    ctx.fillStyle = "#3a2c1c"; ctx.fillRect(180, 250, 220, 40);
  } else if (roomKey === "dorm") {
    ctx.fillStyle = "#3a2c1c"; ctx.fillRect(180, 250, 150, 40); ctx.fillStyle = "#475569"; ctx.fillRect(180, 238, 150, 14);
    ctx.fillStyle = "#0c1322"; _roundRect(ctx, 560, 60, 150, 90, 8); ctx.fill();
    ctx.font = "40px system-ui"; ctx.textAlign = "center"; ctx.fillText("🌙", 635, 118);
  } else if (roomKey === "grad") {
    ctx.fillStyle = "#3a2c1c"; ctx.fillRect(280, 250, 260, 18);
    ctx.font = "30px system-ui"; ctx.textAlign = "center"; ctx.fillText("🎓", 200, 110); ctx.fillText("🎓", 600, 110);
    ctx.fillStyle = "#fbbf24"; ctx.font = "600 16px system-ui"; ctx.fillText("✨ Congratulations ✨", 400, 100);
  } else if (roomKey === "common") {
    ctx.fillStyle = "#3a2c1c"; ctx.fillRect(180, 250, 130, 36); ctx.fillRect(480, 250, 130, 36);
    ctx.font = "34px system-ui"; ctx.textAlign = "center"; ctx.fillText("☕", 400, 110);
  }

  // door + window on the back wall
  ctx.fillStyle = "#0c1118"; ctx.fillRect(36, 96, 42, 114); ctx.fillStyle = "#475569"; ctx.fillRect(70, 150, 5, 8);

  // agents glide to their interior spots
  const spots = ROOM_SPOTS[roomKey] || ROOM_SPOTS.dean;
  ["teacher", "student"].forEach((who) => {
    const ag = CAMPUS.agents[who];
    ag.x += (spots[who][0] - ag.x) * 0.1; ag.y += (spots[who][1] - ag.y) * 0.1;
  });
  _char(ctx, CAMPUS.agents.teacher, "Teacher", 0);
  _char(ctx, CAMPUS.agents.student, "Student", 8);

  // banner + footer + bubble + confetti
  ctx.fillStyle = "#e8eef7"; ctx.font = "600 15px system-ui"; ctx.textAlign = "left";
  ctx.fillText(CAMPUS.banner || "", 14, 26);
  ctx.fillStyle = "#7c8aa0"; ctx.font = "11px system-ui";
  ctx.fillText(`${room.emoji} ${room.name}${CAMPUS.pin ? "  (pinned — Auto-follow off)" : ""}`, 14, H - 12);
  const bub = CAMPUS.bubble;
  if (bub.who && bub.text && CAMPUS.agents[bub.who]) {
    const ag = CAMPUS.agents[bub.who];
    _bubbleOver(ctx, ag.x, ag.y, bub.text, W);
  }
  if (CAMPUS.confetti.length) {
    CAMPUS.confetti.forEach((p) => { p.y += p.vy; ctx.fillStyle = p.c; ctx.fillRect(p.x, p.y, p.s, p.s); });
    CAMPUS.confetti = CAMPUS.confetti.filter((p) => p.y < H);
  }
  if (CAMPUS.glowT > 0) CAMPUS.glowT--;
}

function campusDraw() {
  const ctx = CAMPUS.ctx, cv = ctx.canvas;
  if (!cv.offsetParent) return;
  CAMPUS.frame++;

  // pinned views override the state machine
  if (CAMPUS.pin === "world") { drawWorld(ctx, cv); return; }
  if (CAMPUS.pin) { drawInterior(ctx, cv, CAMPUS.pin); return; }

  if (CAMPUS.cohort) { drawWorld(ctx, cv); return; }   // cohort lives in the world

  if (CAMPUS.mode === "interior") { drawInterior(ctx, cv, CAMPUS.room || "dean"); return; }

  // world mode: walk, then zoom into the target building
  const s = CAMPUS.agents.student, t = CAMPUS.agents.teacher;
  const arrived = !s.waypoints.length && !t.waypoints.length;
  if (CAMPUS.walkTarget && arrived) CAMPUS.zoom = Math.min(1, CAMPUS.zoom + 0.045);
  if (CAMPUS.zoom > 0 && CAMPUS.walkTarget) {
    const b = WORLD.buildings[CAMPUS.walkTarget];
    const z = 1 + CAMPUS.zoom * 1.6;          // scale 1 → 2.6 toward the building
    ctx.save();
    ctx.translate(cv.width / 2, cv.height / 2);
    ctx.scale(z, z);
    ctx.translate(-b.x, -b.y);
    drawWorld(ctx, cv);
    ctx.restore();
    if (CAMPUS.zoom >= 1) {                    // step inside
      CAMPUS.mode = "interior";
      CAMPUS.room = CAMPUS.walkTarget;
      CAMPUS.walkTarget = null;
      CAMPUS.zoom = 0;
      const spots = ROOM_SPOTS[CAMPUS.room] || ROOM_SPOTS.dean;
      s.x = 400; s.y = 300; t.x = 430; t.y = 300;   // enter through the door
      void spots;
    }
    return;
  }
  drawWorld(ctx, cv);
}

const CHAT_AV = { teacher: "🧑‍🏫", student: "🤖", you: "🧑‍🎓" };
const CHAT_WHO = { teacher: "Teacher", student: "Student", you: "You" };
const PERSONA_EMOJI = { diligent: "🤓", balanced: "🙂", social: "😎" };
const CHAT_ROLE = { lesson: "introduces", coursework: "prepares coursework", study: "studies",
  question: "asks", answer: "answers", refine: "revises", critique: "critiques", grade: "grades",
  guide: "guides the teacher", override: "overrides", curriculum: "edits the curriculum",
  thesis: "writes a thesis", lecture: "opens a lecture", attend: "attends the lecture",
  social: "heads to the common area", reveal: "reveals the best answer", learn: "learns from a peer" };

function renderCohortBoard(evs) {
  const el = document.getElementById("cohortBoard");
  const start = evs.find((e) => e.type === "cohort_start");
  if (!el || !start) { if (el) el.hidden = true; return; }
  el.hidden = false;
  const lb = [...evs].reverse().find((e) => e.type === "cohort_leaderboard");
  let rows;
  if (lb) rows = lb.board;
  else {
    const agg = {};
    evs.filter((e) => e.type === "cohort_subject").forEach((e) => e.board.forEach((b) => { (agg[b.name] = agg[b.name] || []).push(b.mastery); }));
    rows = start.students.map((s) => ({ name: s.name, persona: s.persona, emoji: s.emoji,
      gpa: (agg[s.name] && agg[s.name].length) ? agg[s.name].reduce((a, c) => a + c, 0) / agg[s.name].length : 0, skipped: [] }));
    rows.sort((a, b) => b.gpa - a.gpa);
  }
  const fin = evs.find((e) => e.type === "final" && e.cohort);
  const winner = fin ? fin.winner : (rows[0] && rows[0].name);
  el.innerHTML = `<div class="cb-head">🏆 Leaderboard <span class="muted tiny-note">— only the best student is saved at graduation</span></div>` +
    rows.map((r, i) => `<div class="cb-row ${r.name === winner ? "win" : ""}">
      <span class="cb-rank">${i + 1}</span><span>${r.emoji || ""}</span>
      <span class="cb-name">${escapeHtml(r.name)} <span class="muted tiny-note">${escapeHtml(r.persona || "")}</span>${r.name === winner ? " 👑" : ""}</span>
      <span class="cb-bar"><span style="width:${Math.round((r.gpa || 0) * 100)}%"></span></span>
      <span class="cb-gpa">${Math.round((r.gpa || 0) * 100)}%</span>
      ${(r.skipped && r.skipped.length) ? `<span class="muted tiny-note">skipped ${r.skipped.length}</span>` : ""}
    </div>`).join("");
}

function renderCampusChat(chat) {
  const el = document.getElementById("campusChat");
  if (!el || !chat || !chat.length) return;
  let lastQi = null;
  el.innerHTML = chat.slice(-80).map((m) => {
    const sep = (m.qi != null && m.qi !== lastQi && m.role === "question")
      ? `<div class="cc-sep">— Question ${m.qi + 1}${m.subject ? " · " + escapeHtml(m.subject) : ""} —</div>` : "";
    lastQi = m.qi != null ? m.qi : lastQi;
    const score = m.score != null
      ? ` <span class="cc-score ${m.passed === false || m.score < 4 ? "bad" : "ok"}">score ${m.score}/5</span>` : "";
    const isStudent = !!m.persona;          // cohort student message
    const who = CHAT_WHO[m.who] || m.who;
    const av = isStudent ? (PERSONA_EMOJI[m.persona] || "🎓") : (CHAT_AV[m.who] || "•");
    const cls = isStudent ? "student" : m.who;
    // override buttons on a graded (non-user, non-cohort) question
    const ov = (m.role === "grade" && m.question && !m.user)
      ? `<div class="cc-ovr">teacher said <b>${m.verdict}</b> · override:
          <button class="cc-ob" data-q="${escapeHtml(m.question)}" data-v="pass">✓ correct</button>
          <button class="cc-ob" data-q="${escapeHtml(m.question)}" data-v="fail">✗ wrong</button></div>` : "";
    return `${sep}<div class="cc-row cc-${cls}">
      <span class="cc-av">${av}</span>
      <div class="cc-body"><span class="cc-role">${who} ${CHAT_ROLE[m.role] || m.role}${score}</span>
        <div class="cc-text">${escapeHtml(m.text)}</div>${ov}</div></div>`;
  }).join("");
  el.querySelectorAll(".cc-ob").forEach((b) => b.onclick = () =>
    uniIntervene({ action: "override", question: b.dataset.q, verdict: b.dataset.v }, `marked as ${b.dataset.v}`));
  el.scrollTop = el.scrollHeight;
}

async function uniIntervene(body, okMsg) {
  const note = document.getElementById("interveneNote");
  const r = await (await api("/api/uni/intervene", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  if (note) note.textContent = r.ok ? `✓ ${okMsg || "sent to the run"}` : `⚠ ${r.error || "could not apply"}`;
  return r.ok;
}

function _wrapText(ctx, text, maxw) {
  const words = text.split(" "); const lines = []; let line = "";
  words.forEach((w) => {
    const test = line ? line + " " + w : w;
    if (ctx.measureText(test).width > maxw && line) { lines.push(line); line = w; }
    else line = test;
  });
  if (line) lines.push(line);
  return lines.slice(0, 3);
}
// =============================================================================

function initAuto() {
  document.getElementById("autoTarget").oninput = (e) =>
    (document.getElementById("autoTargetV").textContent = Number(e.target.value).toFixed(2));
  campusInit();
  loadUniModels();
  api("/api/uni/status").then((r) => r.json()).then((s) => {
    if (s.active_model) document.getElementById("uniActive").textContent = s.active_model;
  }).catch(() => {});
}

document.getElementById("autoStartBtn").onclick = async () => {
  const body = {
    model: document.getElementById("uniModel").value || "rl-graduate",
    domain: document.getElementById("uniDomain").value,
    target: Number(document.getElementById("autoTarget").value),
    max_subjects: Number(document.getElementById("autoRounds").value),
    max_minutes: Number(document.getElementById("autoMins").value),
    max_budget: Number(document.getElementById("autoBudget").value),
    students: Number(document.getElementById("autoStudents").value) || 1,
    depth: Number(document.getElementById("uniDepth").value) || 18,
    student: document.getElementById("uniStudent").value,
  };
  AUTO.target = body.target; AUTO.model = body.model;
  CAMPUS.cohort = false; CAMPUS.students = []; CAMPUS._conf = false; CAMPUS.confetti = [];
  CAMPUS.mode = "world"; CAMPUS.room = null; CAMPUS.walkTarget = null; CAMPUS.zoom = 0;
  CAMPUS.agents.student.waypoints = []; CAMPUS.agents.teacher.waypoints = [];
  CAMPUS.agents.teacher.dest = null; CAMPUS.agents.student.dest = null;
  document.getElementById("cohortBoard").hidden = true;
  const r = await (await api("/api/uni/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  })).json();
  if (r.error) {
    const st = document.getElementById("autoStatus");
    if (r.can_force && confirm("A run is already marked as in progress (it may be stale from an older session). Force-reset it and start fresh?")) {
      await api("/api/uni/force_reset", { method: "POST" });
      st.textContent = "↻ cleared the old run — press Enrol & teach again.";
    } else {
      st.textContent = "⚠ " + r.error;
    }
    return;
  }
  document.getElementById("autoStartBtn").disabled = true;
  document.getElementById("autoStopBtn").disabled = false;
  document.getElementById("autoSummary").hidden = true;
  document.getElementById("campusChat").innerHTML = `<div class="muted tiny-note">Waiting for the first lesson…</div>`;
  if (AUTO.poll) clearInterval(AUTO.poll);
  AUTO.poll = setInterval(pollUni, 1000);
};

document.getElementById("autoStopBtn").onclick = async () => {
  await api("/api/uni/stop", { method: "POST" });
  document.getElementById("autoStatus").textContent = "stopping after this subject…";
};

document.getElementById("autoForceBtn").onclick = async () => {
  if (!confirm("Force-reset the University run state? Use this to clear a stuck or old run so you can start a new one.")) return;
  const r = await (await api("/api/uni/force_reset", { method: "POST" })).json();
  if (AUTO.poll) { clearInterval(AUTO.poll); AUTO.poll = null; }
  document.getElementById("autoStartBtn").disabled = false;
  document.getElementById("autoStopBtn").disabled = true;
  document.getElementById("autoStatus").textContent = r.was_alive
    ? "↻ reset — the old run is winding down; you can start fresh now."
    : "↻ reset — stale run cleared; ready to enrol.";
};
// --- live interventions ---
document.getElementById("ivGuideBtn").onclick = async () => {
  const el = document.getElementById("ivGuide");
  if (el.value.trim() && await uniIntervene({ action: "guide", text: el.value.trim() }, "guidance queued for the next study round")) el.value = "";
};
document.getElementById("ivAskBtn").onclick = async () => {
  const el = document.getElementById("ivAsk");
  if (el.value.trim() && await uniIntervene({ action: "ask", text: el.value.trim() }, "your question will be put to the student")) el.value = "";
};
document.getElementById("ivAddBtn").onclick = async () => {
  const el = document.getElementById("ivSubj");
  if (el.value.trim() && await uniIntervene({ action: "curriculum", op: "add", subject: el.value.trim() }, "subject added to the plan")) el.value = "";
};
document.getElementById("ivRemBtn").onclick = async () => {
  const el = document.getElementById("ivSubj");
  if (el.value.trim() && await uniIntervene({ action: "curriculum", op: "remove", subject: el.value.trim() }, "subject removed from the plan")) el.value = "";
};

document.getElementById("autoDownloadBtn").onclick = () => { window.location = "/api/uni/log"; };
document.getElementById("uniRefreshBtn").onclick = () => loadUniModels();

// --- test the trained model ---
document.getElementById("uniAskBtn").onclick = async () => {
  const q = document.getElementById("uniQ").value.trim();
  if (!q) return;
  const base = document.getElementById("uniAnsBase"), tuned = document.getElementById("uniAnsTuned");
  const libUsed = document.getElementById("uniLibUsed");
  base.textContent = "thinking…"; tuned.textContent = "thinking…"; libUsed.textContent = "";
  base.classList.add("muted"); tuned.classList.add("muted");
  let r;
  try {
    r = await (await api("/api/uni/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, library: document.getElementById("uniLib").checked,
                             model: document.getElementById("uniModelSel").value }) })).json();
  } catch { base.textContent = "⚠ server error"; tuned.textContent = ""; return; }
  if (r.error) { base.textContent = "⚠ " + r.error; tuned.textContent = ""; return; }
  base.textContent = r.base; base.classList.remove("muted");
  tuned.textContent = r.tuned || "(no trained model loaded yet — train or load one)";
  if (r.tuned) tuned.classList.remove("muted");
  if (r.library_used && r.library_used.length)
    libUsed.textContent = "📚 consulted: " + r.library_used.map((x) => `“${x.slice(0, 60)}”`).join(" · ");
};

document.getElementById("uniActivateBtn").onclick = async () => {
  const model = document.getElementById("uniModelSel").value;
  if (!model) return;
  const note = document.getElementById("uniActiveNote");
  note.textContent = " · loading " + model + " (rebuilding from enabled skills)…";
  const r = await (await api("/api/uni/activate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) })).json();
  if (r.ok === false) { note.textContent = " · ⚠ " + (r.error || "busy"); return; }
  const t = setInterval(async () => {
    const s = await (await api("/api/uni/status")).json();
    if (s.rebuild !== "running") { clearInterval(t); note.textContent = " · ✓ loaded"; document.getElementById("uniActive").textContent = model; }
  }, 1000);
};

async function pollUni() {
  let s;
  try { s = await (await api("/api/uni/status")).json(); } catch { return; }
  renderUniLog(s.events);
  renderSyllabus(s.events);
  renderAutoGraph(s.active || {});
  campusScene(s.events, s.active);
  renderCampusChat(s.chat);
  // the most recent thing anyone said floats over their head in the world
  const lastMsg = (s.chat || [])[s.chat.length - 1];
  if (lastMsg && lastMsg.who !== "you")
    CAMPUS.bubble = { who: lastMsg.who, text: lastMsg.text.slice(0, 110) + (lastMsg.text.length > 110 ? "…" : "") };
  const a = s.active || {};
  const cost = (s.events.filter((e) => e.cost != null).pop() || {}).cost || 0;
  document.getElementById("autoStatus").innerHTML =
    `teaching <b>${a.subject || "…"}</b> · spent <b>$${cost.toFixed(3)}</b>`;
  const nowEl = document.getElementById("autoNow");
  if (a.question) { nowEl.hidden = false; nowEl.innerHTML = `▶ <b>Exam Q${(a.qi ?? 0) + 1}:</b> ${escapeHtml(a.question)}`; }
  else nowEl.hidden = true;
  if (s.active_model) document.getElementById("uniActive").textContent = s.active_model;

  if (s.status === "done" || s.status === "error") {
    clearInterval(AUTO.poll); AUTO.poll = null;
    document.getElementById("autoStartBtn").disabled = false;
    document.getElementById("autoStopBtn").disabled = true;
    if (s.summary) showUniSummary(s.summary);
    loadUniModels();
  }
}

function renderUniLog(events) {
  const el = document.getElementById("autoLog");
  if (!events || !events.length) return;
  const ICON = { start: "🚀", curriculum: "📚", subject_start: "📖", study: "🧑‍🏫", reading: "📖",
                 exam: "✍️", subject_done: "🎓", replan: "🔁", thesis: "📜", stopped: "■", final: "🏁",
                 finals_start: "📝", final_exam: "🧠", general_check: "🌐", error: "⚠" };
  el.innerHTML = events.slice(-90).map((e) => {
    let t = "";
    if (e.type === "start") t = `enrolled '${e.model}' in ${e.domain} · dean: ${e.teacher}${e.is_api ? "" : " (local stand-in)"}`;
    else if (e.type === "curriculum") t = `Dean designed a ${e.subjects.length}-subject curriculum`;
    else if (e.type === "subject_start") t = `→ ${e.subject}${e.why ? " — " + e.why : ""}`;
    else if (e.type === "study") t = `studying ${e.subject} (${e.examples} examples) · $${(e.cost || 0).toFixed(3)}`;
    else if (e.type === "exam") t = `exam ${e.subject}: mastery ${(e.mastery * 100).toFixed(0)}% (W${e.wins}/T${e.ties}/L${e.losses})`;
    else if (e.type === "subject_done") t = `${e.mastered ? "✅ mastered" : "⚠ partial"}: ${e.subject} (${(e.mastery * 100).toFixed(0)}%)`;
    else if (e.type === "replan") t = `Dean re-plans weak subjects: ${e.weak.join(", ")}`;
    else if (e.type === "thesis") t = `📜 ${e.model} wrote a graduation thesis “${e.title}” (${e.words} words) — see the Thesis Library`;
    else if (e.type === "reading") t = `reading period: ${e.subject} chapter (${e.words} words)`;
    else if (e.type === "finals_start") t = `FINALS WEEK — full revision, then cumulative exams over ${e.subjects.length} subjects`;
    else if (e.type === "final_exam") t = `final: ${e.subject} — retained ${(e.retention * 100).toFixed(0)}%`;
    else if (e.type === "general_check") t = `general-ability check: perplexity ${e.ppl_base} → ${e.ppl_tuned} (${e.drift_pct > 0 ? "+" : ""}${e.drift_pct}%)`;
    else if (e.type === "stopped") t = `stopped (${e.reason})`;
    else if (e.type === "final") t = `graduated · GPA ${(e.gpa * 100).toFixed(0)}% · ${e.mastered}/${e.total} mastered · $${(e.cost || 0).toFixed(3)}`;
    else if (e.type === "error") t = e.msg;
    return `<div class="al-row"><span class="al-ic">${ICON[e.type] || "·"}</span>${escapeHtml(t)}</div>`;
  }).join("");
  el.scrollTop = el.scrollHeight;
}

function renderSyllabus(events) {
  const el = document.getElementById("uniSyllabus");
  const cur = [...events].reverse().find((e) => e.type === "curriculum");
  if (!cur) { el.innerHTML = ""; return; }
  // latest mastery per subject from subject_done/exam
  const mastery = {};
  events.forEach((e) => { if (e.type === "exam" || e.type === "subject_done") mastery[e.subject] = e.mastery; });
  const active = (events.filter((e) => e.type === "subject_start").pop() || {}).subject;
  el.innerHTML = `<div class="syl-head">Curriculum</div>` + cur.subjects.map((s) => {
    const m = mastery[s.subject];
    const pct = m != null ? Math.round(m * 100) : 0;
    const cls = m != null && m >= AUTO.target ? "done" : (s.subject === active ? "active" : "");
    return `<div class="syl-row ${cls}">
      <div class="syl-name">${escapeHtml(s.subject)} ${m != null ? `<span class="syl-pct">${pct}%</span>` : ""}</div>
      <div class="syl-bar"><span style="width:${pct}%"></span></div></div>`;
  }).join("");
}

function showUniSummary(s) {
  const el = document.getElementById("autoSummary");
  el.hidden = false;
  const extras = [];
  if (s.retention != null) extras.push(`🧠 retention ${(s.retention * 100).toFixed(0)}%`);
  if (s.general) extras.push(`🌐 general ability ${s.general.drift_pct > 0 ? "+" : ""}${s.general.drift_pct}% ppl${Math.abs(s.general.drift_pct) > 25 ? " ⚠" : ""}`);
  const tail = extras.length ? " · " + extras.join(" · ") : "";
  const head = (s.cohort
    ? `🏆 ${escapeHtml(s.winner)} (${escapeHtml(s.winner_persona || "")}) won the term · GPA ${((s.gpa || 0) * 100).toFixed(0)}% · saved as ${escapeHtml(s.model)}`
    : `🎓 ${escapeHtml(s.model)} graduated · GPA ${((s.gpa || 0) * 100).toFixed(0)}% · ${s.mastered}/${s.total} subjects mastered · $${(s.cost || 0).toFixed(3)}`) + tail;
  const note = s.cohort
    ? "Only the winning student was saved as this model. The full leaderboard is above; its thesis & record are in the library."
    : "Skills saved to this model. Use the Model &amp; skills panel above to toggle or retrain any of them.";
  el.innerHTML = `<div class="as-head">${head}</div>
    <div class="as-prompt">${note}${
      s.thesis_id ? ` <button class="tiny" id="summThesisBtn">📜 Read “${escapeHtml(s.thesis_title || "thesis")}”</button>` : ""}</div>`;
  if (s.thesis_id) document.getElementById("summThesisBtn").onclick = () => showThesis(s.thesis_id);
}

// --- models & skills panel -------------------------------------------------
async function loadUniModels() {
  const data = await (await api("/api/uni/models")).json();
  const sel = document.getElementById("uniModelSel");
  const prev = sel.value || AUTO.model;
  sel.innerHTML = data.models.map((m) => `<option value="${m.model}">${m.model} (${m.skills.length} skills)</option>`).join("")
    || `<option value="">no models yet</option>`;
  if (prev && data.models.some((m) => m.model === prev)) sel.value = prev;
  sel.onchange = () => renderSkillsFor(sel.value);
  if (sel.value) renderSkillsFor(sel.value);
}

async function renderSkillsFor(model) {
  if (!model) return;
  const m = await (await api(`/api/uni/manifest?model=${encodeURIComponent(model)}`)).json();
  const el = document.getElementById("uniSkills");
  if (!m.skills || !m.skills.length) { el.innerHTML = `<span class="muted tiny-note">No skills yet for this model.</span>`; return; }
  el.innerHTML = m.skills.map((s) => {
    const pct = Math.round((s.mastery || 0) * 100);
    const ret = s.retention != null
      ? `<span class="skill-ret" title="Retention: score on the cumulative final exam, after ALL subjects were learned">🧠 ${Math.round(s.retention * 100)}%</span>` : "";
    return `<div class="skill-row">
      <label class="skill-toggle"><input type="checkbox" data-skill="${escapeHtml(s.name)}" ${s.enabled ? "checked" : ""}> </label>
      <div class="skill-main"><b>${escapeHtml(s.name)}</b>
        <div class="syl-bar small"><span style="width:${pct}%"></span></div></div>
      <span class="skill-pct">${pct}%</span>${ret}
      <button class="tiny skill-retrain" data-skill="${escapeHtml(s.name)}">retrain</button>
    </div>`;
  }).join("");
  el.querySelectorAll(".skill-toggle input").forEach((cb) => cb.onchange = () => uniSkill("/api/uni/skill_toggle", model, cb.dataset.skill, cb.checked));
  el.querySelectorAll(".skill-retrain").forEach((b) => b.onclick = () => uniSkill("/api/uni/skill_retrain", model, b.dataset.skill));
}

async function uniSkill(endpoint, model, skill, enabled) {
  const note = document.getElementById("uniRebuild");
  note.textContent = `rebuilding ${model} from enabled skills…`;
  const body = enabled === undefined ? { model, skill } : { model, skill, enabled };
  const r = await (await api(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  if (r.ok === false) { note.textContent = "⚠ " + (r.error || "busy"); return; }
  if (AUTO.rebuildPoll) clearInterval(AUTO.rebuildPoll);
  AUTO.rebuildPoll = setInterval(async () => {
    const s = await (await api("/api/uni/status")).json();
    if (s.rebuild !== "running") { clearInterval(AUTO.rebuildPoll); AUTO.rebuildPoll = null; note.textContent = "✓ model rebuilt from enabled skills"; renderSkillsFor(model); }
  }, 1000);
}

// --- learning record (per-model lesson timeline) ---------------------------
const VERDICT_ICON = { pass: "✅", partial: "🟡", fail: "❌" };

async function showRecord(model) {
  if (!model) return;
  document.getElementById("recordTitle").textContent = `📜 Learning record — ${model}`;
  const body = document.getElementById("recordBody");
  body.innerHTML = `<span class="muted tiny-note">Loading…</span>`;
  document.getElementById("recordModal").hidden = false;
  const [data, thData] = await Promise.all([
    (await api(`/api/uni/record?model=${encodeURIComponent(model)}`)).json(),
    (await api("/api/uni/theses")).json(),
  ]);
  const lessons = data.lessons || [];
  // graduation thesis (if any) shown at the top of the record
  const th = (thData.theses || []).find((t) => t.model === model);
  const thesisBanner = th ? `<div class="rec-thesis">
      <div><b>📜 Graduation thesis:</b> “${escapeHtml(th.title)}” · ${th.words || 0} words · GPA ${Math.round((th.gpa || 0) * 100)}%</div>
      <button class="tiny" id="recThesisBtn" data-id="${escapeHtml(th.id)}">Read thesis</button>
    </div>` : "";
  if (!lessons.length) {
    body.innerHTML = thesisBanner + `<span class="muted tiny-note">No lessons recorded yet. Enrol &amp; teach this model to build its timeline.</span>`;
    if (th) document.getElementById("recThesisBtn").onclick = () => showThesis(th.id);
    return;
  }
  body.innerHTML = thesisBanner + lessons.map((L, i) => {
    const pct = Math.round((L.mastery || 0) * 100);
    const cls = L.mastery >= 0.7 ? "ok" : (L.mastery >= 0.4 ? "warn" : "bad");
    const tag = `lesson ${i + 1}` + (L.attempt != null ? ` · try ${L.attempt + 1}` : "");
    const cw = (L.coursework || []).map((c) =>
      `<div class="rec-qa"><div class="rec-q">📖 ${escapeHtml(c.question)}</div><div class="rec-a">${escapeHtml(c.answer)}</div></div>`).join("")
      || `<span class="muted tiny-note">no new coursework this lesson</span>`;
    const ex = (L.exam || []).map((e) =>
      `<div class="rec-qa rec-${e.verdict}"><div class="rec-q">${VERDICT_ICON[e.verdict] || "•"} ${escapeHtml(e.question)}</div>
        <div class="rec-a"><b>model said:</b> ${escapeHtml(e.answer || "(no answer)")}</div>
        <div class="rec-a rec-gold"><b>reference:</b> ${escapeHtml(e.gold || "")}</div></div>`).join("")
      || `<span class="muted tiny-note">no exam this lesson</span>`;
    return `<details class="rec-lesson" ${i === lessons.length - 1 ? "open" : ""}>
      <summary><span class="rec-tag">${tag}</span> <b>${escapeHtml(L.subject || "")}</b>
        <span class="rec-mastery ${cls}">${pct}% mastery</span>
        <span class="muted tiny-note">W${L.wins || 0}/T${L.ties || 0}/L${L.losses || 0}${L.t != null ? ` · ${L.t}s` : ""}</span></summary>
      ${L.why ? `<div class="muted tiny-note rec-why">${escapeHtml(L.why)}</div>` : ""}
      <div class="rec-sec-label">📚 Coursework studied (${(L.coursework || []).length})</div>${cw}
      <div class="rec-sec-label">📝 Exam — what the model answered</div>${ex}
    </details>`;
  }).join("");
  if (th) document.getElementById("recThesisBtn").onclick = () => showThesis(th.id);
}

document.getElementById("uniRecordBtn").onclick = () => showRecord(document.getElementById("uniModelSel").value);
document.getElementById("recordClose").onclick = () => { document.getElementById("recordModal").hidden = true; };
document.getElementById("recordModal").onclick = (e) => { if (e.target.id === "recordModal") e.target.hidden = true; };

// --- graduation thesis library --------------------------------------------
function mdToHtml(md) {
  return escapeHtml(md || "").split(/\n{2,}/).map((block) => {
    const b = block.trim();
    if (b.startsWith("## ")) return `<h3>${b.slice(3)}</h3>`;
    if (b.startsWith("# ")) return `<h2>${b.slice(2)}</h2>`;
    const t = b.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\*(.+?)\*/g, "<i>$1</i>").replace(/\n/g, "<br>");
    return `<p>${t}</p>`;
  }).join("");
}

async function showThesisLib() {
  const body = document.getElementById("thesisBody");
  document.getElementById("thesisTitle").textContent = "📚 Graduation Thesis Library";
  body.innerHTML = `<span class="muted tiny-note">Loading…</span>`;
  document.getElementById("thesisModal").hidden = false;
  const data = await (await api("/api/uni/theses")).json();
  const ths = data.theses || [];
  if (!ths.length) {
    body.innerHTML = `<span class="muted tiny-note">No theses yet — when a run finishes, the graduating agent writes one and it appears here.</span>`;
    return;
  }
  body.innerHTML = ths.map((t) => `<div class="thesis-card" data-id="${escapeHtml(t.id)}">
     <div class="th-row"><b>${escapeHtml(t.title || t.domain)}</b><span class="th-gpa">GPA ${Math.round((t.gpa || 0) * 100)}%</span></div>
     <div class="muted tiny-note">🎓 ${escapeHtml(t.model)} · ${escapeHtml(t.domain || "")} · ${t.mastered}/${t.total} mastered · ${t.words || 0} words · ${escapeHtml(t.date || "")}</div>
     <div class="muted tiny-note">${escapeHtml((t.specs && t.specs.base_model) || "")} · teacher ${escapeHtml((t.specs && t.specs.teacher) || "")}</div>
   </div>`).join("");
  body.querySelectorAll(".thesis-card").forEach((c) => c.onclick = () => showThesis(c.dataset.id));
}

async function showThesis(id) {
  const body = document.getElementById("thesisBody");
  document.getElementById("thesisModal").hidden = false;   // works when opened directly (e.g. summary button)
  body.innerHTML = `<span class="muted tiny-note">Loading…</span>`;
  const t = await (await api(`/api/uni/thesis?id=${encodeURIComponent(id)}`)).json();
  if (t.error) { body.innerHTML = `⚠ ${t.error}`; return; }
  document.getElementById("thesisTitle").textContent = `📜 ${t.title}`;
  const sp = t.specs || {};
  const rows = { Agent: t.model, Domain: t.domain, "Base model": sp.base_model, Adapter: sp.adapter,
    Teacher: sp.teacher, Device: sp.device, GPA: Math.round((t.gpa || 0) * 100) + "%",
    Mastered: `${t.mastered}/${t.total}`, Date: t.date };
  const specRows = Object.entries(rows).map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(String(v || "—"))}</td></tr>`).join("");
  const cw = t.coursework || {};
  const cwHtml = Object.entries(cw).map(([subj, qa]) => `<details class="th-cw"><summary>${escapeHtml(subj)} — ${qa.length} papers</summary>${
      qa.map((p) => `<div class="rec-qa"><div class="rec-q">📄 ${escapeHtml(p.question)}</div><div class="rec-a">${escapeHtml(p.answer)}</div></div>`).join("")
    }</details>`).join("") || `<span class="muted tiny-note">no coursework stored</span>`;
  body.innerHTML = `<div class="th-actions"><button class="tiny" id="thesisBack">← back to library</button>
      <a class="tiny" href="/api/uni/thesis_download?id=${encodeURIComponent(id)}">⬇ download .md</a></div>
    <div class="thesis-specs"><b>🪪 Agent specifications</b><table>${specRows}</table></div>
    <div class="thesis-doc">${mdToHtml(t.thesis_md)}</div>
    <div class="rec-sec-label">📄 Papers &amp; coursework the student submitted</div>${cwHtml}`;
  document.getElementById("thesisBack").onclick = showThesisLib;
}

document.getElementById("thesisLibBtn").onclick = showThesisLib;
document.getElementById("thesisClose").onclick = () => { document.getElementById("thesisModal").hidden = true; };
document.getElementById("thesisModal").onclick = (e) => { if (e.target.id === "thesisModal") e.target.hidden = true; };

// load an existing model into the enrol form to teach it a NEW topic (keeps its skills)
document.getElementById("uniTeachMoreBtn").onclick = () => {
  const model = document.getElementById("uniModelSel").value;
  if (!model) return;
  document.getElementById("uniModel").value = model;
  const dom = document.getElementById("uniDomain");
  dom.value = "";
  dom.focus();
  dom.scrollIntoView({ behavior: "smooth", block: "center" });
  const note = document.getElementById("uniRebuild");
  note.textContent = `“${model}” loaded — type a new domain above and press Enrol & teach. Existing skills are kept.`;
};

// permanently delete a saved model
document.getElementById("uniDeleteBtn").onclick = async () => {
  const model = document.getElementById("uniModelSel").value;
  if (!model) return;
  if (!confirm(`Delete “${model}” and everything it learned (skills, lessons, transcript)? This cannot be undone.`)) return;
  const note = document.getElementById("uniRebuild");
  note.textContent = `deleting “${model}”…`;
  const r = await (await api("/api/uni/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) })).json();
  if (!r.ok) { note.textContent = "⚠ " + (r.error || "could not delete"); return; }
  note.textContent = `✓ deleted “${model}”`;
  document.getElementById("uniSkills").innerHTML = `<span class="muted tiny-note">Select a model to see its skills.</span>`;
  loadUniModels();
};

// ---------------------------------------------------------------- boot
redrawEditor();   // the maze is drawn client-side; never needs the worker

// Everything else here is served by the worker. In the cloud deployment the
// worker isn't paired at page load, so those fetches would 401 and come back
// empty (no algorithm list, no saved models). connect.js fires 'rl-worker-ready'
// the moment pairing succeeds — we (re)load then, and also re-init whatever view
// is open so its data appears without a manual tab switch.
function loadWorkerData() {
  loadAlgorithms();
  if (state.view && state.view !== "grid") setView(state.view);
}

// Local mode (same-origin) is ready immediately; cloud mode waits for the event.
if (!window.RL || !window.RL.base || window.RL.paired) {
  loadWorkerData();
}
document.addEventListener("rl-worker-ready", loadWorkerData);

