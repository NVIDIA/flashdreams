/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/

/**
 * Homepage "Supported Models" rail (right-hand secondary sidebar).
 *
 * The marketing layout has no pydata page-TOC, so `layout.html` drops an
 * empty `#fd-models-nav` aside on the homepage only. This script reads the
 * on-page "Supported Models" card grid and copies each card's model-page link
 * into the rail.
 *
 * Single source of truth: the grid in index.rst. Add a card and the rail
 * picks it up with no other edits. No-op (and the rail stays hidden) on any
 * page without both the aside and a "Supported Models" section.
 */
(function () {
  "use strict";

  function findModelsSection() {
    // docutils slugifies the "Supported Models" H2 to this section id.
    var byId = document.getElementById("supported-models");
    if (byId) return byId;
    // Fallback: locate the H2 by text, then its enclosing <section>.
    var heads = document.querySelectorAll(".fd-landing-main h2");
    for (var i = 0; i < heads.length; i++) {
      if (heads[i].textContent.trim().replace(/#$/, "").trim() === "Supported Models") {
        return heads[i].closest("section");
      }
    }
    return null;
  }

  function init() {
    var aside = document.getElementById("fd-models-nav");
    if (!aside) return;

    var section = findModelsSection();
    if (!section) return;

    var list = aside.querySelector(".fd-models-nav__list");
    if (!list) return;

    var cols = section.querySelectorAll(".sd-col");
    var count = 0;

    cols.forEach(function (col) {
      var titleEl = col.querySelector(".sd-card-title");
      if (!titleEl) return;
      var label = titleEl.textContent.trim();
      if (!label) return;
      var cardLink = col.querySelector(".sd-stretched-link[href]");
      if (!cardLink) return;

      var li = document.createElement("li");
      li.className = "fd-models-nav__item";
      var a = document.createElement("a");
      a.className = "fd-models-nav__link";
      a.href = cardLink.getAttribute("href");
      a.textContent = label;
      li.appendChild(a);
      list.appendChild(li);
      count += 1;
    });

    if (!count) return;

    aside.hidden = false;
    aside.classList.add("is-ready");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
