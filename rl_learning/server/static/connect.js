/* Where the compute lives.
 *
 * Two deployments, one frontend:
 *
 *   local   The page was served by the Python server itself, so the API is
 *           same-origin. Nothing to configure, no token, no AWS. (Default.)
 *
 *   cloud   The page came from CloudFront and the GPU is on the user's own
 *           machine, behind a loopback address the cloud cannot see. The page
 *           talks *directly* to http://127.0.0.1:5057 — training data, models
 *           and API keys never traverse our servers.
 *
 * Everything below exists to make those two cases look identical to app.js,
 * which just calls api("/api/...") and never knows the difference.
 */

const RL = {
  // Same-origin unless we were loaded from the cloud UI.
  base: "",
  token: localStorage.getItem("rl_worker_token") || "",
  paired: false,
  capabilities: null,
};

// If we are not being served by the Python server, the worker is on loopback.
const IS_CLOUD = !!window.RL_CLOUD_UI;
const DEFAULT_WORKER = localStorage.getItem("rl_worker_url") || "http://127.0.0.1:5057";
if (IS_CLOUD) RL.base = DEFAULT_WORKER;

/** fetch(), but pointed at the worker and carrying the pairing token. */
async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (RL.base && RL.token) headers["Authorization"] = `Bearer ${RL.token}`;

  let res;
  try {
    res = await fetch(RL.base + path, { ...opts, headers });
  } catch (err) {
    // A cross-origin failure to loopback is almost always "no worker running",
    // which is a normal state to be in, not a crash.
    if (RL.base) {
      RL.paired = false;
      renderWorkerStatus();
      throw new Error("No local worker detected. Start it to use the GPU stages.");
    }
    throw err;
  }

  if (res.status === 401) {
    RL.paired = false;
    renderWorkerStatus();
    throw new Error("This browser is not paired with the worker. Paste the pairing token.");
  }
  return res;
}

/** Is a worker alive, and does it accept our token? */
async function probeWorker() {
  if (!RL.base) {                       // local mode: we *are* the worker
    RL.paired = true;
    await loadCapabilities();
    renderWorkerStatus();
    announceReady();
    return true;
  }

  try {
    const health = await (await fetch(RL.base + "/api/health")).json();
    if (!health.ok) throw new Error("bad health");
  } catch {
    RL.paired = false;
    renderWorkerStatus();
    return false;
  }

  // Alive — but the token still has to be right, which /api/capabilities proves.
  try {
    await loadCapabilities();
    RL.paired = true;
  } catch {
    RL.paired = false;
  }
  renderWorkerStatus();
  if (RL.paired) announceReady();
  return RL.paired;
}

async function loadCapabilities() {
  const res = await api("/api/capabilities");
  if (!res.ok) throw new Error("unauthorized");
  RL.capabilities = await res.json();
  return RL.capabilities;
}

// Announce that the worker is usable, so app.js can (re)load the data it
// couldn't fetch before we were paired (algorithm list, saved models, …).
// Fired once per transition into the ready state.
function announceReady() {
  if (RL._announced) return;
  RL._announced = true;
  document.dispatchEvent(new CustomEvent("rl-worker-ready"));
}

/** Store a token the user pasted, then re-check. */
async function pairWorker(token, url) {
  RL.token = (token || "").trim();
  localStorage.setItem("rl_worker_token", RL.token);
  if (url) {
    RL.base = url.replace(/\/$/, "");
    localStorage.setItem("rl_worker_url", RL.base);
  }
  return probeWorker();
}

/** Paint the "GPU connected" pill, and disable the stages that need one. */
function renderWorkerStatus() {
  const el = document.getElementById("workerStatus");
  const caps = RL.capabilities;

  // Lost the worker? Allow a future re-pair to re-announce readiness.
  if (!RL.paired) RL._announced = false;

  if (el) {
    if (!RL.paired) {
      el.className = "worker-pill worker-off";
      el.textContent = RL.base ? "⚠ No worker connected" : "⚠ Worker unavailable";
    } else if (caps && caps.gpu && caps.gpu.available) {
      el.className = "worker-pill worker-on";
      el.textContent = `🟢 GPU: ${caps.gpu.name} (${caps.gpu.vram_gb} GB)`;
    } else {
      el.className = "worker-pill worker-cpu";
      el.textContent = "🟡 Connected — CPU only (no CUDA GPU found)";
    }
  }

  // Deep RL and LLM distillation need VRAM; pathfinding and tabular RL do not.
  const gpuReady = !!(RL.paired && caps && caps.gpu && caps.gpu.available);
  document.querySelectorAll("[data-needs-gpu]").forEach((node) => {
    node.classList.toggle("stage-locked", !gpuReady);
    node.title = gpuReady ? "" : "Connect a machine with a CUDA GPU to use this stage.";
  });
}

/* ── The pairing dialog ──────────────────────────────────────────────────────
 * Only ever shown in the cloud deployment. Locally the worker *is* the origin,
 * so there is nothing to pair and the whole bar stays hidden.
 */
function initPairingUI() {
  const bar = document.getElementById("workerBar");
  const overlay = document.getElementById("pairOverlay");
  if (!bar || !overlay) return;

  if (!IS_CLOUD) {
    bar.hidden = true;               // local: the question doesn't arise
    return;
  }
  bar.hidden = false;

  const tokenInput = document.getElementById("pairToken");
  const urlInput = document.getElementById("pairUrl");
  const errorEl = document.getElementById("pairError");
  const submit = document.getElementById("pairSubmit");

  const open = () => {
    overlay.hidden = false;
    errorEl.hidden = true;
    urlInput.value = RL.base || DEFAULT_WORKER;
    tokenInput.focus();
  };
  const close = () => { overlay.hidden = true; };

  const connect = async () => {
    const token = tokenInput.value.trim();
    if (!token) {
      errorEl.textContent = "Paste the pairing token the worker printed.";
      errorEl.hidden = false;
      return;
    }

    submit.disabled = true;
    submit.textContent = "Connecting…";
    try {
      const ok = await pairWorker(token, urlInput.value.trim());
      if (ok) {
        close();
      } else {
        // Distinguish the two failures the user can actually act on: nothing
        // listening at all, versus listening but rejecting this token/origin.
        const reachable = await fetch(RL.base + "/api/health").then(() => true).catch(() => false);
        errorEl.textContent = reachable
          ? "The worker refused that token. Check you copied it whole, and that it was started with --allow-origin " + window.location.origin + "."
          : "No worker is listening at " + RL.base + ". Is `rl-playground --mode worker` running?";
        errorEl.hidden = false;
      }
    } finally {
      submit.disabled = false;
      submit.textContent = "Connect";
    }
  };

  document.getElementById("workerConnectBtn").addEventListener("click", open);
  document.getElementById("workerStatus").addEventListener("click", open);
  document.getElementById("pairCancel").addEventListener("click", close);
  submit.addEventListener("click", connect);
  tokenInput.addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  // Land on the cloud page with no worker yet? Say so straight away rather than
  // letting the user click Train and get a silent failure.
  if (!RL.token) open();
}

window.api = api;
window.RL = RL;
window.pairWorker = pairWorker;
window.probeWorker = probeWorker;

document.addEventListener("DOMContentLoaded", () => {
  initPairingUI();
  probeWorker();
});
