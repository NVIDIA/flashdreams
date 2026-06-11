# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sphinx configuration for the FlashDreams documentation site.

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

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
        "quickstart/index": ["page-toc"],
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
    # Keep the navbar at depth 1 to only show top-level landings/sections
    "show_nav_level": 1,
}

# Wire the left-sidebar nav-tree only for certain pages.
html_sidebars = {
    "index": [],
    "quickstart/index": [],
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
