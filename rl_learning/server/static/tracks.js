/* The course catalog — the platform's single source of truth for what can be
 * learned here.
 *
 *   track  →  modules (acts)  →  lessons
 *
 * shell.js renders the playground rail from this file; index.html mirrors it
 * as the track cards on the home page. Adding a subject to the platform means
 * adding data here (plus its views/content) — NOT rewriting the navigation.
 *
 * A lesson:
 *   view      the app.js view id this lesson drives (window.setView)
 *   icon      a glyph name from icons.js
 *   needsGpu  requires a CUDA GPU on the worker (everything needs the worker)
 *   learn     anchor of its theory chapter on /learn
 */
(function () {
  "use strict";

  window.RL_TRACKS = [
    {
      id: "rl",
      title: "Reinforcement Learning",
      icon: "cap",
      status: "live",
      tagline: "Search → RL → agentic LLMs, by watching every algorithm work.",
      modules: [
        {
          title: "Search",
          lessons: [
            { view: "grid", n: 1, icon: "grid", title: "Grid maze",
              blurb: "Pathfinding on a maze you draw — BFS, Dijkstra, A*.",
              needsGpu: false, learn: "/learn/rl/pathfinding" },
            { view: "map", n: 2, icon: "route", title: "Real streets",
              blurb: "The very same searches, racing over live OpenStreetMap roads.",
              needsGpu: false, learn: "/learn/rl/pathfinding" },
          ],
        },
        {
          title: "Deep RL",
          lessons: [
            { view: "game", n: 3, icon: "gamepad", title: "Deep RL",
              blurb: "An agent learns Pac-Man & Pong from raw pixels — DQN to PPO.",
              needsGpu: true, learn: "/learn/rl/deep-rl" },
          ],
        },
        {
          title: "Agentic LLMs",
          lessons: [
            { view: "llm", n: 4, icon: "chip", title: "Teach an LLM",
              blurb: "Distil a big teacher into a small model (SFT → DPO / RLAIF).",
              needsGpu: true, learn: "/learn/rl/teach-llm" },
            { view: "reason", n: 5, icon: "loop", title: "Reasoning Lab",
              blurb: "Wire agents into generate → critique → refine loops.",
              needsGpu: true, learn: "/learn/rl/reasoning" },
            { view: "auto", n: 6, icon: "cap", title: "University",
              blurb: "A Dean designs a curriculum and teaches a model to mastery.",
              needsGpu: true, learn: "/learn/rl/university" },
          ],
        },
      ],
    },

    /* In development. Listed so students see where the platform is going;
       flipping one live = writing its lessons + views, not shell surgery. */
    {
      id: "ml",
      title: "ML Fundamentals",
      icon: "flask",
      status: "soon",
      tagline: "Regression to gradient boosting — watch models fit, overfit and generalise.",
    },
    {
      id: "llm-eng",
      title: "LLM Engineering",
      icon: "chip",
      status: "soon",
      tagline: "Tokenisers, attention, RAG and evals — the stack behind every AI product.",
    },
    {
      id: "devops",
      title: "DevOps",
      icon: "server",
      status: "soon",
      tagline: "Containers, CI/CD, IaC and cloud deployment (AWS · Azure · GCP) — ship what you build.",
    },
  ];
})();
