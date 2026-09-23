# Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or pull
requests — all help is appreciated.

## Reporting issues

Open an issue on the
[GitHub repository](https://github.com/Akseli-Ilmanen/ethograph/issues) with a
clear description of the problem and steps to reproduce it. Please:

1) In the top bar, **Help ▸ Print current state**. Share this message along with your error.
2) If you have data loading problems, send some sample data to akseli.ilmanen@gmail.com, so I can test it myself.


## Development installation

To work on ethograph itself, clone the repository and install it in editable
mode with every optional dependency plus the development tools (pytest,
pytest-qt, ruff, pre-commit) and the docs toolchain:

```bash
git clone https://github.com/Akseli-Ilmanen/ethograph.git
cd ethograph
conda create -y -n ethograph-dev -c conda-forge python=3.12
conda activate ethograph-dev
uv pip install --torch-backend=auto torch torchvision
uv pip install -e ".[gui,audio,model,dandi,dev,docs]"
pre-commit install
```

conda only creates the environment; ethograph itself is installed with uv.
PyTorch is installed first, separately, so that uv picks the right build for
your GPU. If you only want the GUI and its tests, `".[gui,audio,dev]"` is
enough. Then check everything works:

```bash
# downloads all template datasets for the real-data integration tests (can be quite slow)
python -c "from ethograph.datasets import DATASETS; from ethograph.utils.download import ensure_template_dataset; [ensure_template_dataset(k) for k in DATASETS]"
pytest                        # unit + integration tests
pre-commit run --all-files    # what CI runs; run twice, the first pass fixes in place
```

See {doc}`../getting_started/installation` for details on setting up a virtual
environment and installing uv.

## Pull requests

Please open pull requests as **drafts** rather than ready-for-review. Draft
PRs make it easy to discuss the approach early, before time goes into
polishing. Mark it ready for review once tests pass, `pre-commit` is clean
and you consider it finished.

(target-add-your-example)=
## Add your workflow/dataset example!

If you develop a workflow with ethograph that others could re-use (a dataset
conversion, a feature-extraction recipe, a training setup), please share it. Any of these works:

- **Open a pull request** that adds your Jupyter notebook to the
  [`examples/` folder](https://github.com/Akseli-Ilmanen/ethograph/tree/main/examples).
  It is picked up by the docs build and shown in the {doc}`../examples/index`.
- **Email the notebook** to akseli.ilmanen@gmail.com and I will add it for you.
- **Open a GitHub issue** first if you would like to discuss it a bit before
  writing it up.
