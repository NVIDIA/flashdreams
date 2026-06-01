/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/

/**
 * Hover-to-zoom for documentation images and videos (opt-in).
 *
 * Only elements inside a .zoomable container (or with the class directly)
 * participate. On mouseenter the element's full-resolution version is shown
 * in a fixed overlay. The overlay never captures pointer events -- the
 * cursor stays on the source element, so mouseleave fires reliably.
 */
(function () {
  "use strict";

  var overlay = null;
  var clone = null;
  var activeEl = null;
  var showTimer = null;
  var SHOW_DELAY = 120; // ms before showing (avoids flash on quick pass-through)

  function createOverlay() {
    overlay = document.createElement("div");
    overlay.className = "img-zoom-overlay";
    document.body.appendChild(overlay);
  }

  function show(el) {
    if (!overlay) createOverlay();
    if (activeEl === el) return;
    activeEl = el;

    // Remove previous clone
    if (clone && clone.parentNode) clone.parentNode.removeChild(clone);

    if (el.tagName === "VIDEO") {
      clone = document.createElement("video");
      clone.src = el.currentSrc || el.src;
      clone.poster = el.poster || "";
      clone.autoplay = true;
      clone.loop = true;
      clone.muted = true;
      clone.playsInline = true;
      // Sync playback position
      clone.currentTime = el.currentTime;
    } else {
      clone = document.createElement("img");
      clone.src = el.currentSrc || el.src;
      clone.alt = el.alt || "";
    }
    clone.className = "img-zoom-clone";
    overlay.appendChild(clone);

    // Force reflow then show
    void overlay.offsetHeight;
    overlay.classList.add("visible");
  }

  function hide() {
    clearTimeout(showTimer);
    showTimer = null;
    if (!overlay) return;
    overlay.classList.remove("visible");
    activeEl = null;
    // Clean up clone after fade-out
    setTimeout(function () {
      if (clone && clone.parentNode && !overlay.classList.contains("visible")) {
        clone.parentNode.removeChild(clone);
        clone = null;
      }
    }, 350);
  }

  function isZoomable(el) {
    var node = el;
    while (node && node !== document.body) {
      if (node.classList && node.classList.contains("zoomable")) return true;
      node = node.parentElement;
    }
    return false;
  }

  function isZoomTarget(el) {
    return (el.tagName === "IMG" || el.tagName === "VIDEO") && isZoomable(el);
  }

  function init() {
    var content = document.querySelector(".content") ||
                  document.querySelector("[role='main']") ||
                  document.body;

    content.addEventListener("mouseenter", function (e) {
      var target = e.target;
      if (isZoomTarget(target)) {
        clearTimeout(showTimer);
        showTimer = setTimeout(function () {
          show(target);
        }, SHOW_DELAY);
      }
    }, true);

    content.addEventListener("mouseleave", function (e) {
      var target = e.target;
      if (target === activeEl || (showTimer && isZoomTarget(target))) {
        hide();
      }
    }, true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
