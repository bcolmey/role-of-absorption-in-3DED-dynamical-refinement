The role of absorption in 3D electron diffraction dynamical structure refinement
==================================================================================

**Acta Crystallographica Section A: Foundations and Advances**, Volume 82, Part 6, November 2026 (open access)
ISSN 2053-2733 | https://doi.org/10.1107/S2053273326009459

Benjamin Colmey<sup>a,*</sup>, Tiarnan A. S. Doherty<sup>a,b</sup>, Shreshth A. Malik<sup>b</sup> and Paul A. Midgley<sup>a,*</sup>

<sup>a</sup> Department of Materials Science and Metallurgy, University of Cambridge, 27 Charles Babbage Road, Cambridge, CB3 0FS, United Kingdom
<sup>b</sup> OATML, Department of Computer Science, University of Oxford, Wolfson Building, Parks Road, Oxford, OX1 3QG, United Kingdom

<sup>*</sup> Correspondence e-mail: bc626@cam.ac.uk, pam33@cam.ac.uk

Received 10 February 2026; accepted 9 September 2026; online 21 September 2026.

This repository holds the simulation code, configuration files and source data for the paper.


Abstract
--------

The role of absorption in 3D electron diffraction is established through analytical theory, simulation and dynamical
refinement. A two-beam expression for the absorbed integrated intensity in centrosymmetric crystals is derived, showing
that for t/&xi;<sub>g</sub> &ll; 1 reflections follow a uniform exponential decay set by the mean absorptive potential
U<sub>0</sub>&prime;. Many-beam simulations of both centrosymmetric and non-centrosymmetric crystals reveal additional
reflection-specific anomalous absorption beyond the uniform attenuation set by U<sub>0</sub>&prime;. Neglecting these
effects in dynamical refinement of integrated intensities incurs an error that increases approximately linearly with
thickness, with this error becoming more severe near zone axes. Dynamical refinements were performed on CsPbBr<sub>3</sub>,
quartz and borane, with the inclusion of absorption yielding an improvement in R<sub>obs</sub> from 6.4 to 5.3% for
CsPbBr<sub>3</sub>, and negligible improvements for quartz and borane. Anomalous absorption may therefore be ignored for
routine refinement of integrated intensities except in high-Z materials at thicknesses approaching &xi;<sub>g</sub>.

**Keywords:** absorption; electron diffraction; thermal diffuse scattering; 3D ED; dynamical diffraction.

Supporting information: https://doi.org/10.1107/S2053273326009459/lu5052sup1.pdf


Citation
--------

```bibtex
@article{Colmey2026,
  author  = {Colmey, Benjamin and Doherty, Tiarnan A. S. and Malik, Shreshth A. and Midgley, Paul A.},
  title   = {The role of absorption in 3{D} electron diffraction dynamical structure refinement},
  journal = {Acta Crystallographica Section A: Foundations and Advances},
  volume  = {82},
  number  = {6},
  year    = {2026},
  month   = nov,
  doi     = {10.1107/S2053273326009459}
}
```


Data Sources
------------

The three experimental 3D ED datasets refined in the paper come from:

- CsPbBr3 and borane (B18H22): Suresh, A. *et al.* (2024). Nat. Commun. 15, 9066. https://doi.org/10.1038/s41467-024-53448-2
- &alpha;-quartz: Klar, P. B. *et al.* (2023). Nat. Chem. 15, 848-855.

The Bloch-wave refinement framework is described in Malik, S. A. *et al.* (2026). Nat. Commun. 17, 5056.


Repository Layout
-----------------

This repository is organized around three main directories:

- `source_notebooks/`
  Paper-facing analysis notebooks used to reproduce figures and source-data processing.

- `absorption_paper_source_data/`
  Curated source data used by the notebooks. 

- `diffBloch_version_0.0.1/`
  The `diffBloch` simulation code, configuration files, crystallographic input files.


Source Notebooks
----------------

The notebooks in `source_notebooks/` are the main entry points for the paper analysis:

- `two_beam.ipynb`
  Two-beam absorption calculations, Lorentz-corrected two-beam comparisons, many-beam comparison plots, and related thickness/decay analyses.

- `Lorentz_correction.ipynb`
  Determination of Lorentz factors for CsPbBr3.

- `R1_vs_thickness.ipynb`
  R1/residual analysis as a function of thickness.

- `rocking_curve_plotting.ipynb`
  Rocking-curve inspection.

- `Non_centro.ipynb`
  Non-centrosymmetric Ge/GaAs absorption and structure-factor diagnostics.

- `refinement_results_visualization.ipynb`
  Plots and tables comparing elastic/no-absorption and absorptive refinement outputs.


Source Data
-----------

`absorption_paper_source_data/` contains the processed data products used directly by the notebooks:

- `two_beam/`
  Two-beam CSV inputs and outputs, including Ug tables, Lorentz corrections, and Lorentz-corrected two-beam intensities.

- `many_beam_simulations/`
  Many-beam intensity and rocking-curve CSVs for borane, CsPbBr3, and quartz.

- `refinement_results/`
  Refinement summaries and exported tables for absorptive and no-absorption models. Each material/model folder contains:
  `asu.csv`, `atp.csv`, `residuals.csv`, and `thickness.csv`.
  Summary CIFs are stored alongside these folders.

- `non_centro/`
  Ge/GaAs non-centrosymmetric simulation outputs and structure-factor diagnostic tables.

- `cspbbr3/outputs/reciprocal_frames/`
  Generated reciprocal-space/Ewald-frame images used for CsPbBr3 visualization.


Environment 

Create the environment and install the local package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```
