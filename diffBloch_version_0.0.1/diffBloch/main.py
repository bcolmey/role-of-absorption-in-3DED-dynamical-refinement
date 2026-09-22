"""Forward-only inference entry point for the paper reproduction.

Usage (after `pip install -e .` from the repository root):

    infer quartz-absorption
    infer quartz-no-abs
    infer cspbbr3-absorption inference.rotations=[0,1,2,3,4,5]
    python -m diffBloch.main program=inference experiment=borane-no-abs

There is one config per model in configs/experiment/ (quartz, cspbbr3, borane, each with -absorption
and -no-abs).

Only inference is supported. Nothing here builds an optimizer or calls backward();
the whole run executes inside torch.inference_mode().

Thickness is read per orientation from an exported thickness-versus-tilt-angle curve
(`inference.thickness_curve`, columns Theta, Pred Thickness), or is the single flat value
`inference.thickness` when no curve is set (`inference.thickness_curve=null`). The thickness
network itself is never evaluated.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"
BASE_CONFIGS = ("atoms", "bloch", "refinement", "structure_factor", "thicknessNN")


def _parse_cli(argv: list[str]) -> tuple[str, str | None, list[str]]:
    program, experiment, overrides = "inference", None, []
    for item in argv:
        if "=" not in item:
            if experiment is not None:
                raise SystemExit(f"Unexpected argument: {item}")
            experiment = item  # bare experiment name
            continue
        key, value = item.split("=", 1)
        if key == "program":
            program = value
        elif key == "experiment":
            experiment = value
        else:
            overrides.append(item)
    return program, experiment, overrides


def load_config(experiment: str, overrides: list[str]):
    experiment_path = CONFIG_ROOT / "experiment" / f"{experiment}.yaml"
    if not experiment_path.exists():
        available = ", ".join(p.stem for p in sorted((CONFIG_ROOT / "experiment").glob("*.yaml")))
        raise SystemExit(f"Unknown experiment '{experiment}'. Available: {available}")

    cfg = OmegaConf.create({name: OmegaConf.load(CONFIG_ROOT / name / "base.yaml") for name in BASE_CONFIGS})
    cfg = OmegaConf.merge(cfg, OmegaConf.load(experiment_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # Experiment files use ${hydra:runtime.cwd} for the package root and ${repo_root}
    # for the repository root; resolve both without needing hydra at runtime.
    text = OmegaConf.to_yaml(cfg, resolve=False)
    text = text.replace("${hydra:runtime.cwd}", str(PACKAGE_ROOT)).replace("${repo_root}", str(REPO_ROOT))
    return OmegaConf.create(text)


def _load_thickness_curve(cfg):
    """Exported thickness curve as (theta_degrees, thickness_angstrom) arrays, or None to use the flat value."""
    path = cfg.inference.get("thickness_curve", None)
    if not path:
        return None
    curve = pd.read_csv(path)
    if "Step" in curve.columns:
        curve = curve[curve["Step"] == curve["Step"].max()]  # final exported step
    curve = curve.drop_duplicates(subset="Theta").sort_values("Theta")  # some exports repeat the same curve
    return curve["Theta"].to_numpy(), curve["Pred Thickness"].to_numpy()


def _text_table(headers: list[str], rows: list[list[str]]) -> str:
    """Fixed-width text table with a dashed rule under the header."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)])


def _rule(title: str) -> str:
    return f"--- {title} " + "-" * max(4, 78 - len(title) - 5)


def _write_report(path: Path, cfg, atoms, structure_factors, dataset, metrics, curve, elapsed: float) -> None:
    """Human-readable report of one inference run (parameters, residuals, per-orientation table, structure used)."""
    from ase.data import chemical_symbols

    cell = np.asarray(atoms.unit_cell, dtype=float)
    a, b, c = np.linalg.norm(cell, axis=1)
    ang = lambda u, v: float(np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))))
    match = re.search(r"#(\d+) \(([^,)]+)", str(atoms.spacegroup))  # e.g. "SpaceGroup #154 (P3221, Trigonal). ..."
    sg_text = f"{match.group(2)} (#{match.group(1)})" if match else str(atoms.spacegroup)
    d, r = cfg.refinement.data, cfg.bloch
    thickness = (f"exported curve, {curve[1].min():.0f}-{curve[1].max():.0f} A" if curve is not None
                 else f"flat, {cfg.inference.thickness} A")
    params = [
        ("Integration semiangle (deg)", d.integration_semiangle),
        ("Rocking-curve sampling", d.rocking_curve_sampling),
        ("D_sg", d.dsg),
        ("R_sg", d.rsg),
        ("g_max, beam cutoff (A^-1)", r.g_max),
        ("structure-factor table radius (A^-1)", f"{structure_factors.g_max:g}  (2 x g_max + 0.5)"),
        ("g_max refine, resolution cut (A^-1)", r.g_max_refine),
        ("sg_max (A^-1)", r.sg_max),
        ("Absorption (T/F)", "T" if cfg.structure_factor.absorption else "F"),
        ("Absorption type", cfg.structure_factor.absorption_type if cfg.structure_factor.absorption else "-"),
        ("Electron energy (keV)", f"{structure_factors.energy / 1e3:g}"),
        ("Thickness", thickness),
        ("Orientations (simulated)", len(metrics)),
    ]
    out = [
        "=" * 78,
        f" diffBloch inference report -- {cfg.inference.name}",
        "=" * 78,
        f" mode        : forward-only inference (no refinement, no optimisation)",
        f" structure   : {Path(cfg.atoms.data.cif_file_path).name}",
        f" exp_data    : {Path(cfg.refinement.data.pets_path).name}",
        f" orientations: {Path(cfg.bloch.optim_orientations_path).name}",
        f" generated   : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f" elapsed     : {elapsed:.1f} s ({elapsed / 60:.2f} min)",
        "",
        _rule("Simulation parameters"),
        _text_table(["Parameter", "Value"], [[k, str(v)] for k, v in params]),
        "",
        _rule("Crystallographic parameters"),
        _text_table(["Parameter", "Value"], [
            ["Space group", sg_text],
            ["a, b, c (A)", f"{a:.4f}, {b:.4f}, {c:.4f}"],
            ["alpha, beta, gamma (deg)", f"{ang(cell[1], cell[2]):.2f}, {ang(cell[0], cell[2]):.2f}, {ang(cell[0], cell[1]):.2f}"],
            ["Volume (A^3)", f"{float(atoms.cell_volume()):.1f}"],
            ["N atoms (ASU)", str(len(atoms.asu_positions))],
        ]),
        "",
        _rule("Residuals (mean over orientations)"),
    ]
    rows = [["This run", f"{metrics['rbragg'].mean() * 100:.2f}", f"{metrics['wr'].mean() * 100:.2f}"]]
    out += [_text_table(["", "R_obs (%)", "wR (%)"], rows), ""]
    out += [_rule("Per-orientation residuals")]
    headers = ["Rotation", "Thickness (A)", "R_obs", "wR", "N matched"]
    rows = []
    for idx, m in metrics.iterrows():
        row = [str(idx), f"{m['thickness']:.1f}", f"{m['rbragg']:.6f}", f"{m['wr']:.6f}", str(int(m["num_bragg_spots"]))]
        rows.append(row)
    out += [_text_table(headers, rows), ""]
    out += [_rule("Structure used (not refined here) -- asymmetric unit, fractional coordinates")]
    pos = atoms.asu_positions.detach().cpu().numpy()
    rows = [[str(lbl), chemical_symbols[int(z)], *(f"{x:.6f}" for x in xyz)]
            for lbl, z, xyz in zip(atoms.asu_atom_labels, atoms.asu_numbers.detach().cpu().numpy(), pos)]
    out += [_text_table(["Label", "Element", "x", "y", "z"], rows), ""]
    out += [_rule("Files"), " metrics.csv  per-orientation numbers", " config.yaml   the fully resolved settings of this run", ""]
    path.write_text("\n".join(out))


def run_inference(cfg) -> Path:
    if not cfg.refinement.inference_only:
        raise RuntimeError("Inference refused: refinement.inference_only must be true")
    started = time.time()

    cfg.bloch.thicknesses = [float(cfg.inference.thickness)]
    cfg.bloch.optim_thicknesses_path = None
    cfg.thicknessNN.activate = False  # the network is never evaluated
    curve = _load_thickness_curve(cfg)

    from diffBloch.atoms import Atoms
    from diffBloch.dynamical import ApparentThicknessNN, BlochNet, StructureFactorNet
    from diffBloch.rotation_dataset import get_dataloaders
    from diffBloch.utils import initialize_scaling_factor, resolution_filter_diffraction_intensities

    print(f"experiment={cfg.inference.name} "
          f"absorption={cfg.structure_factor.absorption}/{cfg.structure_factor.absorption_type} "
          + (f"thickness from curve ({curve[1].min():.0f}-{curve[1].max():.0f} A)" if curve is not None
             else f"flat thickness={cfg.inference.thickness} A"))
    print(f"orientations: {cfg.bloch.optim_orientations_path}")

    with torch.inference_mode():
        _, dataloader = get_dataloaders(
            cfg.refinement,
            default_thickness=list(cfg.bloch.thicknesses),
            optim_orientations_path=cfg.bloch.optim_orientations_path,
            optim_thicknesses_path=None,
        )
        dataset = dataloader.dataset
        alpha_deg = dataset.alpha_scaler.inverse_transform(np.asarray(dataset.alphas).reshape(-1, 1)).ravel()
        atoms = Atoms(cfg.atoms)
        thickness_nn = ApparentThicknessNN(cfg.thicknessNN)
        structure_factors = StructureFactorNet(cfg.structure_factor, atoms, thickness_nn=thickness_nn, solve_g_max=cfg.bloch.g_max)
        model = BlochNet(cfg.bloch, structure_factors, refine_vg=False)
        model.eval()

        ignored = set(cfg.refinement.dataloader.ignore_orientations)
        if cfg.inference.rotations is None:
            rotations = [i for i in range(len(dataset)) if i not in ignored]
        else:
            rotations = [int(i) for i in cfg.inference.rotations]
        tilts = dataset.rocking_curve_orientations
        mosaicity = dataset.mosaicity_num_frames if cfg.refinement.data.mosaicity else None

        rows = []
        for r in rotations:
            if r + 1 not in dataset.exp_info:
                print(f"rotation {r}: no experimental reflections, skipped")
                continue
            _, rotation, _, thickness = dataset[r]
            if curve is not None:
                thickness = torch.tensor([float(np.interp(alpha_deg[r], curve[0], curve[1]))], dtype=torch.float32)
            model.Fgb = model.structure_factor_net()
            result = model(orientation_matrix=rotation, tilts=tilts, thickness=thickness)
            result.filter_hkls(
                energy=model.energy,
                rsg=cfg.refinement.data.rsg,
                dsg=cfg.refinement.data.dsg,
                semiangle=cfg.refinement.data.integration_semiangle,
            )
            intensities, hkls = result.get_integrated_intensities(mosaicity=mosaicity)
            exp, sigma, sim, matched_hkls = result.compare_experimental_simulated_data(
                dataset.exp_info[r + 1], hkls, intensities[-1]
            )
            exp, sigma, sim, _ = resolution_filter_diffraction_intensities(
                exp, sigma, sim, matched_hkls, cfg.bloch.g_max_refine, cfg.bloch.g_min_refine, atoms.reciprocal_cell()
            )
            if len(exp) == 0:
                print(f"rotation {r}: no reflections after resolution filter, skipped")
                continue
            _, rbragg = initialize_scaling_factor(exp, sim, sigma, r_value_method="rbragg_abs")
            _, wr = initialize_scaling_factor(exp, sim, sigma, r_value_method="wrbragg")
            t = float(thickness.flatten()[0])
            rows.append({"rotation_idx": r, "thickness": t, "rbragg": float(rbragg), "wr": float(wr), "num_bragg_spots": len(exp)})
            print(f"rotation {r}: R(obs)={float(rbragg):.6f} wR={float(wr):.6f} n_refl={len(exp)} t={t:.0f}", flush=True)

    if not rows:
        raise RuntimeError("Inference completed without any matched reflections")

    metrics = pd.DataFrame(rows).set_index("rotation_idx")
    print("\n" + metrics.round(6).to_string())
    print(f"\nMean R(obs): {metrics['rbragg'].mean():.6f}    Mean wR: {metrics['wr'].mean():.6f}    ({len(metrics)} orientations)")

    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.csv"
    metrics.to_csv(output_path)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    report_path = output_dir / "inference_report.txt"
    _write_report(report_path, cfg, atoms, structure_factors, dataset, metrics, curve, time.time() - started)
    print(f"Saved: {output_path}")
    print(f"Saved: {report_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    program, experiment, overrides = _parse_cli(sys.argv[1:] if argv is None else argv)
    if program != "inference":
        raise SystemExit("This bundle supports only inference; refinement is disabled")
    if not experiment:
        raise SystemExit("usage: infer <experiment> [key=value ...]   (e.g. infer quartz-absorption inference.rotations=[0,1])")
    run_inference(load_config(experiment, overrides))


if __name__ == "__main__":
    main()
