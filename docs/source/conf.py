# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sphinx configuration for the FlashDreams documentation site.

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

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

html_theme = "pydata_sphinx_theme"
html_title = f"FlashDreams {version}"
html_show_sphinx = False
html_static_path = ["_static"]
html_logo = "../../assets/logo/flashdreams_logo_horizontal.png"

html_theme_options = {
    "secondary_sidebar_items": ["page-toc"],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "github_url": "https://github.com/NVIDIA/flashdreams",
    # Drop the "Built with the PyData Sphinx Theme x.y.z" footer credit.
    "footer_start": ["copyright"],
    "footer_end": [],
    "navigation_depth": 4,
    # `False` so each section's sub-pages stay visible in the left
    # sidebar when you're on a page within that section. `True`
    # collapses all sub-trees regardless of current page, which on
    # this site leaves the sidebar showing only the same seven
    # top-level entries the navbar already carries.
    "collapse_navigation": False,
    # -- Top navbar arrangement -----------------------------------------
    # Logo | centered nav | theme switcher + GitHub icon (auto-rendered
    # from `github_url`) | persistent search button.
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    # Keep the navbar at depth 1 — only the master_doc's top-level
    # toctree entries appear there. Sub-pages live in each section's
    # own toctree, which drives the left sidebar instead.
    "show_nav_level": 1,
    # Seven top-level sections in the master toctree (Benchmarks, four
    # docs captioned trees, two ref captioned trees, Community); promote
    # them all into the primary navbar instead of bucketing into "More".
    "header_links_before_dropdown": 7,
}

# Wire the left-sidebar nav-tree component explicitly. Without this,
# pydata renders the primary sidebar container (with the "Collapse
# Sidebar" toggle) but no nav contents. The homepage, benchmarks page,
# and community pages are front-of-site / marketing surfaces — give
# them no sidebar; every reference-docs page gets the section nav tree.
html_sidebars = {
    "index": [],
    "benchmarks/*": [],
    "community/*": [],
    "getting_started/*": ["sidebar-nav-bs"],
    "developer_guides/*": ["sidebar-nav-bs"],
    "models/*": ["sidebar-nav-bs"],
    "apis/*": ["sidebar-nav-bs"],
    "reference/*": ["sidebar-nav-bs"],
}

html_context = {
    "github_user": "NVIDIA",
    "github_repo": "flashdreams",
    "github_version": "main",
    "doc_path": "docs/source",
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
    # `flashdreams.core.attention.rope_kernel` does `import triton` at
    # module load. CPU-only torch wheels don't ship triton, so the
    # docs-ci env can't import it — mock to keep autodoc building
    # without pulling the 176MB wheel onto the runner.
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
