/* Theme boot + the canvas bridge.
 *
 * Load this IN <head>, before the stylesheet paints anything: it stamps the
 * saved theme choice on <html> so the first paint is already the right theme
 * (no flash). Three states:
 *
 *   (nothing stored)      follow the OS via prefers-color-scheme (CSS only)
 *   rl_theme = "light"    <html data-theme="light">
 *   rl_theme = "dark"     <html data-theme="dark">
 *
 * It also exposes window.RLTheme — how the CANVAS renderers (maze, race cards,
 * training curves) read the same CSS tokens the rest of the UI uses. Canvas 2D
 * can't use var(--x), so RLTheme.get("--cell") resolves it once and caches;
 * the cache is dropped and "rl-theme-change" is dispatched whenever the theme
 * flips (toggle click or OS change), so listeners can redraw.
 */
(function () {
  "use strict";

  var KEY = "rl_theme";

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function apply(choice) {
    if (choice === "light" || choice === "dark") {
      document.documentElement.setAttribute("data-theme", choice);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }
  apply(stored());

  var cache = {};
  function effective() {
    var s = stored();
    if (s === "light" || s === "dark") return s;
    return window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches
      ? "light" : "dark";
  }

  window.RLTheme = {
    /** Resolve a CSS custom property (e.g. "--cell") to its current value. */
    get: function (name) {
      if (cache[name]) return cache[name];
      var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      if (v) cache[name] = v;
      return v || null;
    },
    current: effective,
    toggle: function () {
      var next = effective() === "light" ? "dark" : "light";
      try { localStorage.setItem(KEY, next); } catch (e) { /* private mode */ }
      apply(next);
      changed();
    },
  };

  function changed() {
    cache = {};
    document.dispatchEvent(new CustomEvent("rl-theme-change", { detail: { theme: effective() } }));
  }

  // Following the OS (no explicit choice) means reacting when the OS flips.
  if (window.matchMedia) {
    matchMedia("(prefers-color-scheme: light)").addEventListener("change", function () {
      if (!stored()) changed();
    });
  }

  // Any element with [data-theme-toggle] flips the theme (the topbar button).
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
      btn.addEventListener("click", window.RLTheme.toggle);
    });
  });
})();
