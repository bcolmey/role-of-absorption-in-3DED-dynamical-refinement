# Colmey et al. (2026), Acta Crystallographica A 82(6), November 2026


This directory holds the diffBloch code (version 0.0.1), configs and inputs needed to reproduce the
three-material comparison in the paper, elastic and absorptive, by forward simulation:

```text
configs/experiment/quartz-absorption.yaml    configs/experiment/quartz-no-abs.yaml
configs/experiment/cspbbr3-absorption.yaml   configs/experiment/cspbbr3-no-abs.yaml
configs/experiment/borane-absorption.yaml    configs/experiment/borane-no-abs.yaml
```

Each one is run by name: `infer quartz-absorption`, `infer quartz-no-abs`, and so on.

For speed, this bundle runs the forward simulation only. Each `infer` run simulates every experimental orientation with the
optimised orientations and structure in `data/`, with absorption or without, and reports R_obs and wR per orientation,
saved to `inference_outputs/<experiment>/`.

**Run time on a CPU:** a quartz model takes a few minutes. CsPbBr3 and borane take considerably longer, borane the longest. This is how long it took running on a CPU, before the latest updates to diffBloch, including speedups and GPU optimisation. A full refinement (a forward and a backward pass over every orientation, for tens of epochs) would take days on the same CPU, for this reason we provide `infer`, which runs the forward simulation only and reproduces the with- and without-absorption residuals of the archived refined models without repeating the refinement.

## Data source

CsPbBr3 and borane data:

> Suresh, A., Yörük, E., Cabaj, M. K., Brázda, P., Výborný, K., Sedláček, O., Müller, C.,
> Chintakindi, H., Eigner, V. & Palatinus, L. (2024). *Ionisation of atoms determined by kappa
> refinement against 3D electron diffraction data.* Nature Communications 15, 9066.
> https://doi.org/10.1038/s41467-024-53448-2

alpha-quartz data:

> Klar, P. B., Krysiak, Y., Xu, H., Steciuk, G., Cho, J., Zou, X. & Palatinus, L. (2023).
> Nature Chemistry 15, 848-855.

```bibtex
@article{Suresh2024,
  author  = {Ashwin Suresh and Emre Yörük and Małgorzata K. Cabaj and Petr Brázda and
             Karel Výborný and Ondřej Sedláček and Christian Müller and
             Hrushikesh Chintakindi and Václav Eigner and Lukáš Palatinus},
  title   = {Ionisation of atoms determined by kappa refinement against 3D electron diffraction data},
  journal = {Nature Communications},
  volume  = {15},
  pages   = {9066},
  year    = {2024},
  doi     = {10.1038/s41467-024-53448-2},
  url     = {https://doi.org/10.1038/s41467-024-53448-2}
}
```
## Running infer

Set up the environment once, from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

Absorptive and elastic runs for one material (here quartz):

```bash
infer quartz-absorption     # absorptive: absorption on, absorptive orientation set
infer quartz-no-abs         # elastic: absorption off, elastic orientation set
```

Each run prints R_obs, wR and the reflection count for every orientation, and saves three files in `inference_outputs/<experiment>/`:

- `inference_report.txt`: the run's report, in the style of a refinement report. It has the simulation and crystallographic
  parameters, mean R_obs and wR, a per-orientation table (thickness used, R_obs, wR, matched reflections), and the structure used.
- `metrics.csv`: the same per-orientation numbers as a table.
- `config.yaml`: the fully resolved settings of that run.

