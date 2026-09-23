# Developer commands

Everything in one place: environment, tests, lint, docs, release.
All commands run from the repo root in the `ethograph` conda env:

```powershell
(C:\Users\aksel\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate ethograph)
```

## Install

```powershell
uv pip install -e ".[gui,audio,dandi,dev,docs]"
```

Also the fix when `ethograph.__version__` shows `0.1.dev...` — reinstalling picks up the latest tag.

```powershell
python -c "import ethograph; print(ethograph.__version__)"
```

## Tests

Layout: `tests/test_unit/` (Qt-free logic or one widget — fast) and `tests/test_integration/` (full window + a dataset).
Files prefixed `_test_` are ad-hoc debug scripts and are skipped by pytest.
Config lives in `pyproject.toml` `[tool.pytest.ini_options]`: `--cov=ethograph -m 'not slow'` is always on, and every warning is an error.

```powershell
pytest                                             # everything except slow (network / >30 s)
pytest tests/test_unit                             # fast suite
pytest tests/test_unit/test_curation.py            # one file
pytest tests/test_unit/test_curation.py -k Active  # one class / test by name
pytest -x --no-cov -q                              # stop at first failure, no coverage, terse
pytest -m slow                                     # only the slow ones (needs the datasets)
pytest tests/test_integration --show               # show the GUI window for 15 s after each test
pytest --lf                                        # rerun only what failed last time
```

`pytest_configure` fetches BirdPark and Moll2025 on first run; every other template is skipped (`is_dataset_downloaded`) until downloaded, via the GUI cover page or once for all of them:

```powershell
python -c "from ethograph.datasets import DATASETS; from ethograph.utils.download import ensure_template_dataset; [ensure_template_dataset(k) for k in DATASETS]"
```

What earns a test, which fixture to take: see `CLAUDE.md` → *What earns a test*.

## Lint, format, type check

`pre-commit` is what CI runs (ruff, ruff-format, codespell, check-manifest, whitespace/EOL fixers):

```powershell
pre-commit run --all-files     # run twice: first pass fixes in place, second confirms clean
pre-commit install             # once per clone — runs on every commit
ruff check . ; ruff format .   # just ruff, without the rest
mypy                           # informational (non-blocking in CI); config in pyproject.toml
```

## Docs (Sphinx + MyST)

Two ways to look at the docs — pick one, never both:

**1. Quick look** — build once, open the HTML. Sphinx is incremental: only pages whose
source changed are rebuilt, so this is also how you check a single page.

```powershell
sphinx-build -b html docs/source docs/build/html
start docs/build/html/index.html                   # or any page: docs/build/html/advanced/metadata.html
```

**2. Live editing** — serves at http://127.0.0.1:8000, rebuilds and reloads the browser on every save.
Leave it running in its own terminal while you write.

```powershell
sphinx-autobuild docs/source docs/build/html --open-browser
```

Only when something looks wrong:

```powershell
sphinx-build -b html -E docs/source docs/build/html   # stale toctree / sidebar / cached notebook → rebuild from scratch
sphinx-build -b html -n docs/source docs/build/html   # nitpicky: list every broken {func}/{doc} reference
```

MyST cheat-sheet:

````markdown
{doc}`loading_script`                          # sibling page
{func}`~ethograph.utils.xr_utils.sel_valid`    # Python function (~ hides the module path)
{class}`~pynapple.TsdFrame`                    # external class via intersphinx
{attr}`~pynwb.file.NWBFile.subject`

::::{tab-set}
:::{tab-item} Tab title
Content.
:::
::::

```{note}
A note.        # also {warning}, {important}, {tip}
```
````

Find an intersphinx target: `python -m sphinx.ext.intersphinx https://pynwb.readthedocs.io/en/stable/objects.inv | grep Subject`

| Setting | Location |
|---------|----------|
| Extensions, theme, intersphinx | `docs/source/conf.py` |
| Doc dependencies | `pyproject.toml` → `[project.optional-dependencies]` `docs` |
| Notebook execution | `nb_execution_mode = "off"` in `conf.py` |
| Custom CSS | `docs/source/_static/css/custom.css` |

## Screenshots for the docs

`scripts/screenshot_gui.py` launches the GUI with two extra shortcuts bound, for capturing
figures at a resolution the docs can actually use. Setting `QT_SCALE_FACTOR` is the wrong
tool here — it lays the window out in bigger pixels, and the saved dock layout (sized in the
old pixels) collapses into overlapping panels. This works the other way round: the window
keeps its on-screen geometry and is *re-rendered* into a pixmap at N× the device pixel
ratio, so text and plot curves are redrawn as vectors while every widget stays where it is.
The pygfx video canvases live on the GPU and never appear in a Qt repaint, so each is
snapshotted through its own renderer at the same scale and composited back into place.

```powershell
python scripts/screenshot_gui.py                                   # 3x into ./screenshots
python scripts/screenshot_gui.py --scale 4 --out docs/source/_static/media
python scripts/screenshot_gui.py --scale max                       # largest that fits
```

| Shortcut | What it does |
|----------|--------------|
| `Ctrl+Shift+S` | Capture the whole window to a timestamped PNG in `--out` |
| `Ctrl+Shift+M` | Toggle minimal mode |

**Minimal mode** strips the chrome a figure does not need and enlarges what a reader looks
at: plot titles and axis label text go (tick labels stay — they are data), every panel
header goes, the bottom bar's Proxy checkbox goes, and the menu bar, playback bar and the
Data / Labels / Trials buttons all grow. It is a reversible view over the running session —
every property is saved before it is touched and restored verbatim, no `app_state` is
written, and nothing survives the toggle.

`--scale` is bounded by `max_scale()`: Qt's raster limit, the GPU's maximum texture
dimension (read off the renderer's own device) and a 512 MB RAM budget, which in practice
binds first. An over-large request is clamped and reported rather than taking the process
down — a maximised window at 20× wants ~2.5 GB. Past 4× the extra detail is mostly
invisible, and screen-space-sized things (1 px gridlines, fixed markers) only get thinner;
capture at 3–4× and downscale. For plot panels alone, prefer pyqtgraph's own vector export
(right-click a plot → Export → SVG) over any raster capture.

## Release (PyPI)

Publishing is `.github/workflows/test_and_deploy.yml`, triggered by a **`v*` git tag** — never by a plain push.
Pipeline: lint → manifest → tests → build sdist/wheels → upload to PyPI (3–5 min).

**One-time setup:** PyPI token in `C:\Users\aksel\.pypirc` (outside the repo, never committed):

```ini
[pypi]
username = __token__
password = pypi-your-token-here
```

### The one-liner: `scripts/release.ps1`

```powershell
Unblock-File -Path scripts/release.ps1        # first time on a new PC
scripts/release.ps1                           # commit message defaults to "fix: linting"
scripts/release.ps1 -Message "feat: video grid"
```

It does, in order: `pre-commit run --all-files` → `git add -A` → commit → push `main` → bump the **patch** of the latest `v*` tag (`v0.1.8` → `v0.1.9`) → push the tag. Stops on the first error.
Run pre-commit yourself first if the tree is dirty, so the auto-fix pass does not abort the script.

### By hand (minor/major bump, or when you want to look before tagging)

```powershell
pre-commit run --all-files                    # twice if the first pass changed files
git add -A && git commit -m "..." && git push origin main
git describe --tags --long --match "v*"       # v0.1.8-5-gabcd123 → on v0.1.8
git tag v0.2.0 && git push origin v0.2.0
```

### Monitor

| What | URL |
|------|-----|
| CI progress | https://github.com/Akseli-Ilmanen/ethograph/actions |
| PyPI page | https://pypi.org/project/ethograph/ |
