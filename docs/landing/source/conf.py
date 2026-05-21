# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sphinx configuration for the FlashDreams documentation site.

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Top-navbar links into the reference-docs Sphinx project (separate
# build under /docs/). In CI, `DOCS_SUBDIR` is set to e.g. `main` or
# `versions/0.5.0` and the docs are deployed at
# `<site>/<DOCS_SUBDIR>/docs/`, so a root-relative URL is the only form
# that resolves from every page depth. Local dev (no env var) falls
# back to a depth-0 relative form, which works from the landing root
# only — cross-project links 404 from depth-1 pages locally.
_docs_subdir = os.environ.get("DOCS_SUBDIR", "").strip("/")
_docs_root = f"/{_docs_subdir}/docs/" if _docs_subdir else "docs/"
_docs_nav = [
    {"name": "Getting Started", "url": f"{_docs_root}getting_started/index.html"},
    {"name": "Developer Guides", "url": f"{_docs_root}developer_guides/index.html"},
    {"name": "Models", "url": f"{_docs_root}models/index.html"},
    {"name": "CLI Reference", "url": f"{_docs_root}reference/cli.html"},
    {"name": "API Reference", "url": f"{_docs_root}apis/index.html"},
]

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

# Suppress MyST cross-reference warnings for `.. include::`d markdown that
# uses GitHub-relative file links (e.g. CONTRIBUTING.md links to `LICENSE`,
# `reuse.toml`). Those resolve fine when GitHub renders the file standalone
# but have no analog in the Sphinx build.
suppress_warnings = ["myst.xref_missing"]

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

html_theme = "nvidia_sphinx_theme"
html_title = f"FlashDreams {version}"
html_show_sphinx = False
html_static_path = ["_static"]

html_theme_options = {
    "secondary_sidebar_items": ["page-toc"],
    "copyright_override": {"start": 2026},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "github_url": "https://github.com/NVIDIA/flashdreams",
    "navigation_depth": 4,
    "collapse_navigation": True,
    # -- Top navbar arrangement -----------------------------------------
    # Logo | centered nav | icon links (GitHub auto-renders from
    # github_url) | persistent search button.
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    # Sidebar starts at depth 2: depth-1 (the section names) are already
    # in the navbar; the sidebar should show ONLY the current section's
    # sub-pages, not duplicate the navbar's section list.
    "show_nav_level": 2,
    # The landing toctree promotes Benchmarks and Community to the
    # navbar. Five external links route into the reference-docs project.
    # Bump `header_links_before_dropdown` to 7 (2 toctree + 5 external)
    # so none get bucketed into a "More" dropdown.
    "header_links_before_dropdown": 7,
    "external_links": _docs_nav,
}

# Hide the primary (left) sidebar on marketing-style pages. Every page
# in the landing project uses that layout, so this covers them all.
html_sidebars = {
    "index": [],
    "benchmarks/**": [],
    "community/**": [],
}

html_context = {
    "github_user": "NVIDIA",
    "github_repo": "flashdreams",
    "github_version": "main",
    "doc_path": "docs/landing/source",
    "default_mode": "light",
}

html_css_files = ["custom.css"]

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
