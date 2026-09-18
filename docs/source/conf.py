# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys
from pathlib import Path

# Used when building API docs, put the dependencies
# of any class you are documenting here
autodoc_mock_imports = [
    "audioio",
    "av",
    "crowsetta",
    "dandi",
    "h5py",
    "jinja2",
    "lindi",
    "matplotlib",
    "movement",
    "natsort",
    "neo",
    "noisereduce",
    "numpy",
    "pandas",
    "phylib",
    "pygfx",
    "pynapple",
    "pynaviz",
    "pynwb",
    "pyqtgraph",
    "qtpy",
    "remfile",
    "rendercanvas",
    "ruptures",
    "scipy",
    "sounddevice",
    "soundfile",
    "torch",
    "torchvision",
    "tqdm",
    "vocalpy",
    "librosa",
    "wgpu",
    "xarray",
    "yaml",
]

# Add the module path to sys.path here.
sys.path.insert(0, os.path.abspath("../.."))

project = "ethograph"
copyright = "2026, Akseli Ilmanen"
author = "Akseli Ilmanen"
release = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
    "myst_nb",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.bibtex",
]

# A diagram takes its natural height at the column's width; the extension's
# default squeezes every diagram into a 500px box, which shrinks a tall
# flowchart until its text is unreadable.
mermaid_height = "auto"

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "plain"

# Don't execute notebooks during build
nb_execution_mode = "off"

# Configure the myst parser to enable cool markdown features
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Automatically generate stub pages for API
autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_default_flags = ["members", "inherited-members"]

# List of patterns to ignore when looking for source files.
exclude_patterns = [
    "**.ipynb_checkpoints",
    "**/includes/**",
    "api_generated/**",
    "advanced/examples.md",
    "_static/media/changepoints.ipynb",
]

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pynapple": ("https://pynapple.org/", None),
    "movement": (
        "https://movement.neuroinformatics.dev/latest/",
        None,
    ),
    "pynwb": ("https://pynwb.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "ethograph"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/mermaid_zoom.js"]

# Serve page sources under their own extension rather than the default
# ".txt", so the "Download source" link on an example page hands back a
# usable ``.ipynb`` instead of ``foo.ipynb.txt``.
html_sourcelink_suffix = ""

html_theme_options = {
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Akseli-Ilmanen/ethograph",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "logo": {
        "image_light": "_static/media/icon.png",
        "image_dark": "_static/media/icon.png",
        "text": project,
    },
}

# GitHub pages
github_user = "Akseli-Ilmanen"
html_baseurl = f"https://{github_user}.github.io/{project}"
sitemap_url_scheme = "{link}"





def _remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def setup(app):
    # Notebooks live under examples/ (repo root) and reference their preview
    # images relative to that location, e.g. "../docs/source/_static/media/x.png".
    # The copies below live one level deeper (docs/source/examples/), so that
    # relative prefix is rewritten to "../_static/media/" on copy.
    src = Path(app.srcdir).resolve().parent.parent / "examples"
    dst = Path(app.srcdir) / "examples"
    dst.mkdir(exist_ok=True)

    keep_paths: set[Path] = {dst / "index.rst", dst / ".gitignore"}

    for nb in src.glob("*.ipynb"):
        # Fix double extensions like foo.ipynb.ipynb -> foo.ipynb
        name = nb.name
        while name.endswith(".ipynb.ipynb"):
            name = name[: -len(".ipynb")]
        target = dst / name
        _remove_if_exists(target)
        text = nb.read_text(encoding="utf-8")
        text = text.replace("../docs/source/_static/media/", "../_static/media/")
        target.write_text(text, encoding="utf-8")
        keep_paths.add(target)

    # Remove stale generated files from prior runs.
    for child in dst.iterdir():
        if child in keep_paths:
            continue
        _remove_if_exists(child)
