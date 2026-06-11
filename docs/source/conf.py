# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sphinx configuration for the FlashDreams documentation site.

import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from sphinx.search import languages as _search_languages
from sphinx.search.en import SearchEnglish

# Ensure autodoc imports the in-repo package (flashdreams/flashdreams/*)
# instead of any older site-packages install missing newer modules.
_DOCS_SOURCE_DIR = Path(__file__).resolve().parent
_REPO_SRC_ROOT = _DOCS_SOURCE_DIR.parent.parent / "flashdreams"
sys.path.insert(0, str(_REPO_SRC_ROOT))

# -- Project information -----------------------------------------------------

project = "FlashDreams"
copyright = "2026, NVIDIA Corporation & Affiliates"
author = "NVIDIA"

try:
    release = _pkg_version("flashdreams")
except PackageNotFoundError:
    release = "0.0.0"

# Pretty-print numeric versions (0.1.0 -> v0.1.0).
version = release if release[:1].isalpha() else f"v{release}"

# -- General configuration ---------------------------------------------------

# Treat warnings as errors so broken references / malformed docstrings are
# caught early (locally and in CI).
warningiserror = True

# Auto-generate anchors for markdown headings up to H3 so cross-references
# like `[Project governance](#project-governance)` resolve when MD is
# included via `.. include:: ... :parser: myst_parser.sphinx_`.
myst_heading_anchors = 3

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "torch": ("https://pytorch.org/docs/main/", None),
}
intersphinx_disabled_domains = ["std"]

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"FlashDreams {version}"
html_show_sphinx = False
html_static_path = ["_static", "../../assets/logo"]

html_theme_options = {
    # Light/dark logo split.
    "logo": {
        "image_light": "_static/horizontal-light.svg",
        "image_dark": "_static/horizontal-dark.svg",
    },
    # Google Analytics (GA4) measurement ID.
    "analytics": {
        "google_analytics_id": "G-Q44TKZ8777",
    },
    # Map of pages to secondary sidebar items.
    # Marketing-layout pages have no sidebar and therefore no secondary sidebar items.
    "secondary_sidebar_items": {
        "index": [],
        "quickstart/*": [],
        "community/*": ["page-toc"],
        "community/index": [],
        "models/*": ["page-toc"],
        "models/index": [],
        "documentation": ["page-toc"],
        "developer_guides/*": ["page-toc"],
        "api/*": ["page-toc"],
    },
    # Pygments styles for light/dark mode.
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    # Channel icons for GitHub + Discord
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/flashdreams",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Discord",
            "url": "https://discord.com/invite/nvidiaomniverse",
            "icon": "fa-brands fa-discord",
            "type": "fontawesome",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["icon-links"],
    "navigation_depth": 4,
    "collapse_navigation": False,
    # Top navbar arrangement
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher"],
    "navbar_persistent": ["search-button"],
    "show_nav_level": 2,
}

# Wire the left-sidebar nav-tree only for certain pages.
html_sidebars = {
    "index": [],
    "quickstart/*": [],
    "community/*": ["sidebar-nav-bs"],
    "community/index": [],
    "models/*": ["sidebar-nav-bs"],
    "documentation": ["search-field", "sidebar-nav-bs"],
    "developer_guides/*": ["search-field", "sidebar-nav-bs"],
    "api/*": ["search-field", "sidebar-nav-bs"],
}

html_context = {
    "github_user": "NVIDIA",
    "github_repo": "flashdreams",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "light",
}

html_css_files = ["custom.css"]
html_js_files = ["js/image_zoom.js", "js/supported_models_nav.js"]

# -- Copybutton --------------------------------------------------------------

# Strip Python REPL prompts and shell prompts when copying snippets.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Search: keep hyphenated terms intact ------------------------------------
#
# Sphinx's stock search tokenizer (``\w+``) splits on every non-word character,
# so a hyphenated command like ``flashdreams-run`` is indexed *and* queried as
# the two unrelated words ``flashdreams`` + ``run``. Searching the command then
# matches every page mentioning either half and buries the real hit.
#
# We register an English search language whose tokenizer keeps the hyphenated
# token while still emitting its parts: ``flashdreams-run`` matches the command
# as a cohesive, high-ranked term, and a bare ``flashdreams`` query keeps
# working. The JS query splitter (Sphinx embeds it verbatim as ``splitQuery``
# in language_data.js, overriding the stock one in searchtools.js) mirrors the
# Python indexer so both sides tokenize identically.

class _SearchEnglishHyphenated(SearchEnglish):
    _word_re = re.compile(r"\w+(?:-\w+)*")

    js_splitter_code = r"""
var splitQuery = (query) => {
  const tokens = [];
  const re = /[\p{Letter}\p{Number}_\p{Emoji_Presentation}]+(?:-[\p{Letter}\p{Number}_\p{Emoji_Presentation}]+)*/gu;
  for (const term of query.match(re) || []) {
    tokens.push(term);
    if (term.includes("-"))
      for (const part of term.split("-")) if (part) tokens.push(part);
  }
  return tokens;
};
"""

    def split(self, input: str) -> list[str]:
        tokens: list[str] = []
        for token in self._word_re.findall(input):
            tokens.append(token)
            if "-" in token:
                tokens.extend(part for part in token.split("-") if part)
        return tokens

_search_languages["en"] = _SearchEnglishHyphenated

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

# Don't prepend the full module path to every name in the rendered output.
add_module_names = False

# Many flashdreams modules import torch / transformer-engine at import time.
# Mock the heaviest C-extensions so the docs can build on a CPU-only host
# without the full GPU stack.
autodoc_mock_imports = [
    "transformer_engine",
    "transformer_engine_torch",
    "pynvml",
    "boto3",
    "botocore",
    "mediapy",
    "cv2",
    "triton",
]

# -- Napoleon ----------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
napoleon_custom_sections = [
    ("Phases", "params_style"),
    ("Per-step usage", "params_style"),
    ("Multi-GPU contract", "notes_style"),
    ("Supports", "notes_style"),
    ("Typical usage example", "example_style"),
]
