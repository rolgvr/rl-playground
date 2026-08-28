/* The platform's icon set: one inline SVG sprite, injected once.
 *
 * Replaces the emoji that used to serve as icons (▦ 🗺 🎮 🧠 🔁 🎓 …), which
 * rendered differently on every OS and couldn't follow the theme. These are
 * simple 24×24 stroke glyphs drawn in currentColor, so they inherit text color
 * and work in both themes.
 *
 * Use from HTML:   <svg class="ic"><use href="#i-grid"></use></svg>
 * Use from JS:     icon("grid")            -> the same markup as a string
 *                  icon("grid", "ic-lg")   -> with an extra class
 *
 * Deliberately NOT replaced: the emoji *inside* the simulations (campus rooms,
 * agent avatars, Pac-Man). Those are game content, not chrome.
 */
(function () {
  "use strict";

  var GLYPHS = {
    /* tracks + lessons */
    grid:   '<path d="M4 4h16v16H4z"/><path d="M4 10h16M4 16h16M10 4v16M16 4v16"/>',
    route:  '<circle cx="6" cy="18" r="2.6"/><circle cx="18" cy="6" r="2.6"/><path d="M8.2 16.4C12 13.5 12 10.5 15.8 7.6"/>',
    gamepad:'<path d="M6.5 8h11a4.5 4.5 0 0 1 4.4 5.5l-.8 3.4a2.8 2.8 0 0 1-4.9 1.1L14.6 16H9.4l-1.6 2a2.8 2.8 0 0 1-4.9-1.1l-.8-3.4A4.5 4.5 0 0 1 6.5 8z"/><path d="M8 11v3M6.5 12.5h3"/><path d="M16 11.4h.01M18 13.4h.01"/>',
    chip:   '<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M10 7V4M14 7V4M10 20v-3M14 20v-3M7 10H4M7 14H4M20 10h-3M20 14h-3"/>',
    loop:   '<path d="M4 12a8 8 0 0 1 14-5.3"/><path d="M18 3v4h-4"/><path d="M20 12a8 8 0 0 1-14 5.3"/><path d="M6 21v-4h4"/>',
    cap:    '<path d="M12 4 2 9l10 5 10-5-10-5z"/><path d="M6 11.5V16c0 1.7 2.7 3 6 3s6-1.3 6-3v-4.5"/><path d="M22 9v5"/>',
    book:   '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 20.5V5.5M20 18v3H6.5"/>',
    flask:  '<path d="M9 3h6M10 3v5.2L4.8 17.4A2 2 0 0 0 6.6 21h10.8a2 2 0 0 0 1.8-3.6L14 8.2V3"/><path d="M7.5 15h9"/>',
    server: '<rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><path d="M7 7.5h.01M7 16.5h.01"/>',
    /* chrome + controls */
    play:   '<path d="M7 4.5 19 12 7 19.5z"/>',
    stop:   '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    save:   '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v5h7V3M8 21v-7h8v7"/>',
    download:'<path d="M12 3v11"/><path d="m7 10 5 5 5-5"/><path d="M4 21h16"/>',
    refresh:'<path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 3v4h-4"/>',
    trash:  '<path d="M4 7h16M10 7V4h4v3M6 7l1 14h10l1-14"/><path d="M10 11v6M14 11v6"/>',
    close:  '<path d="m6 6 12 12M18 6 6 18"/>',
    plus:   '<path d="M12 5v14M5 12h14"/>',
    minus:  '<path d="M5 12h14"/>',
    check:  '<path d="m5 13 4 4L19 7"/>',
    search: '<circle cx="11" cy="11" r="6.5"/><path d="m16 16 5 5"/>',
    bolt:   '<path d="M13 2 4.5 13.5H11L10 22l8.5-11.5H13z"/>',
    plug:   '<path d="M9 3v5M15 3v5"/><path d="M6.5 8h11l-1 5a4.5 4.5 0 0 1-4.5 4 4.5 4.5 0 0 1-4.5-4z"/><path d="M12 17v4"/>',
    lock:   '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V7.5a4 4 0 0 1 8 0V11"/>',
    dice:   '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h.01M15 9h.01M12 12h.01M9 15h.01M15 15h.01"/>',
    eraser: '<path d="m7 20-4-4L14.5 4.5a2 2 0 0 1 2.8 0l3.2 3.2a2 2 0 0 1 0 2.8L11 20z"/><path d="M7 20h14"/><path d="m9 8 7 7"/>',
    wall:   '<path d="M3 5h18v14H3z"/><path d="M3 12h18M11 5v3.5M8 12v3.5M15 12v3.5M11 15.5V19"/>',
    droplet:'<path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"/>',
    flag:   '<path d="M6 21V4"/><path d="M6 5c4-2 7 2 12 0v8c-5 2-8-2-12 0"/>',
    target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.4"/>',
    scroll: '<path d="M7 3h12v15a3 3 0 0 1-3 3H7"/><path d="M7 3a2.5 2.5 0 0 0 0 5h2V3zM7 21a2.5 2.5 0 0 1 0-5h9"/><path d="M11 8h5M11 12h5"/>',
    scale:  '<path d="M12 4v16M4 20h16"/><path d="m7 7-3 6h6zM17 7l-3 6h6z"/><path d="M7 7h10"/>',
    teach:  '<path d="M3 4h18v11H3z"/><path d="M8 8h8M8 11h5"/><path d="M9 19l3-4 3 4"/>',
    sun:    '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5V5M12 19v2.5M2.5 12H5M19 12h2.5M5 5l1.7 1.7M17.3 17.3 19 19M19 5l-1.7 1.7M6.7 17.3 5 19"/>',
    moon:   '<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z"/>',
    warn:   '<path d="M12 3 2.5 20h19z"/><path d="M12 9.5V14M12 17h.01"/>',
    github: '<path d="M9 20.5c-5 1.5-5-2.5-7-3m14 6v-3.5c0-1 .1-1.5-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C7.4 5.3 6.4 5.6 6.4 5.6a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 5 12c0 4.6 2.7 5.7 5.5 6-.6.5-.6 1-.5 2V23"/>',
  };

  var sprite = '<svg xmlns="http://www.w3.org/2000/svg" style="display:none" aria-hidden="true">';
  for (var name in GLYPHS) {
    sprite += '<symbol id="i-' + name + '" viewBox="0 0 24 24">' + GLYPHS[name] + "</symbol>";
  }
  sprite += "</svg>";

  function inject() {
    var host = document.createElement("div");
    host.innerHTML = sprite;
    document.body.insertBefore(host.firstChild, document.body.firstChild);
  }
  if (document.body) inject();
  else document.addEventListener("DOMContentLoaded", inject);

  /** Icon markup for JS-built templates. Unknown names render nothing. */
  window.icon = function (name, cls) {
    if (!GLYPHS[name]) return "";
    return '<svg class="ic' + (cls ? " " + cls : "") + '" aria-hidden="true"><use href="#i-' + name + '"></use></svg>';
  };
})();
