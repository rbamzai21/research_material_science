# Research: Material Science

Perovskite stability descriptor research, reproducing and extending Bartel et al. (Sci. Adv. 2019).

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/rbamzai21/research_material_science.git

# If already cloned without submodules
git submodule update --init --recursive

# Python environment (requires uv: https://docs.astral.sh/uv/)
uv sync --all-extras

# Pre-commit hooks
uv run pre-commit install
```

## Project structure

```
├── perovskite-stability/      # Submodule: Bartel et al. data & code
├── reproduce_evidence/         # Scripts reproducing paper figures
│   ├── evidence1_abx3_classification.py   # Fig 2 A, B, C
│   ├── evidence2_double_perovskites.py    # ICSD double perovskites
│   └── evidence3_dft_correlation.py       # Fig 2 D
├── bartel2019-new-tolerance-factor.md     # Paper summary & notes
├── pyproject.toml
└── .pre-commit-config.yaml
```
