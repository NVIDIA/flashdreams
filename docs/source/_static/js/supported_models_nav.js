/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/

/**
 * Homepage "Supported Models" rail (right-hand secondary sidebar).
 *
 * The marketing layout has no pydata page-TOC, so `layout.html` drops an
 * empty `#fd-models-nav` aside on the homepage only. This script reads the
 * on-page "Supported Models" card grid, gives each card a stable anchor id,
 * builds one link per card in the rail, and wires a scroll-spy so the link
 * for the card nearest the top stays highlighted.
 *
 * Single source of truth: the grid in index.rst. Add a card and the rail
 * picks it up with no other edits. No-op (and the rail stays hidden) on any
 * page without both the aside and a "Supported Models" section.
 */
(function () {
  "use strict";

  function slugify(text) {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }

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
    var links = [];
    var seen = {};

    cols.forEach(function (col) {
      var titleEl = col.querySelector(".sd-card-title");
      if (!titleEl) return;
      var label = titleEl.textContent.trim();
      if (!label) return;

      var base = "model-" + slugify(label);
      var id = base;
      var n = 2;
      while (seen[id]) {
        id = base + "-" + n++;
      }
      seen[id] = true;
      col.id = id;
      col.classList.add("fd-model-anchor");

      var li = document.createElement("li");
      li.className = "fd-models-nav__item";
      var a = document.createElement("a");
      a.className = "fd-models-nav__link";
      a.href = "#" + id;
      a.textContent = label;
      li.appendChild(a);
      list.appendChild(li);
      links.push({ link: a, target: col });
    });

    if (!links.length) return;

    aside.hidden = false;
    aside.classList.add("is-ready");

    // Smooth in-page jump (respects reduced-motion via CSS scroll-behavior).
    links.forEach(function (entry) {
      entry.link.addEventListener("click", function () {
        // Defer so the highlight tracks the post-scroll position.
        window.setTimeout(function () {
          setActive(entry.link);
        }, 0);
      });
    });

    function setActive(activeLink) {
      links.forEach(function (entry) {
        var on = entry.link === activeLink;
        entry.link.classList.toggle("is-active", on);
        if (on) {
          entry.link.setAttribute("aria-current", "true");
        } else {
          entry.link.removeAttribute("aria-current");
        }
      });
    }

    // Scroll-spy: highlight the card whose top is nearest just below the
    // sticky header. IntersectionObserver with a top-biased root margin.
    if ("IntersectionObserver" in window) {
      var visible = {};
      var observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (e) {
            visible[e.target.id] = e.isIntersecting;
          });
          for (var i = 0; i < links.length; i++) {
            if (visible[links[i].target.id]) {
              setActive(links[i].link);
              return;
            }
          }
        },
        { rootMargin: "-20% 0px -70% 0px", threshold: 0 }
      );
      links.forEach(function (entry) {
        observer.observe(entry.target);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
