/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/

/**
 * Click-to-zoom for documentation images and videos (opt-in).
 *
 * Only elements inside a .zoomable container (or with the class directly)
 * participate. Clicking the element shows its full-resolution version in a
 * fixed overlay. Clicking the zoomed overlay (or pressing Escape) zooms back
 * out.
 */
(function () {
  "use strict";

  var overlay = null;
  var clone = null;
  var activeEl = null;

  function createOverlay() {
    overlay = document.createElement("div");
    overlay.className = "img-zoom-overlay";
    // Clicking anywhere on the zoomed overlay closes it.
    overlay.addEventListener("click", hide);
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

    content.addEventListener("click", function (e) {
      var target = e.target;
      if (isZoomTarget(target)) {
        e.preventDefault();
        show(target);
      }
    });

    // Pressing Escape zooms back out.
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && activeEl) hide();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
