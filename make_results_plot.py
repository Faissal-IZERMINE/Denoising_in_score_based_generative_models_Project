"""Generate the FID/IS comparison plot used in the README.

All numbers are transcribed from Tables 1-3 of REPORT.pdf.
Run once with ``python make_results_plot.py`` to refresh ``assets/results.png``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# (dataset, baseline FID, hybrid FID, baseline IS, hybrid IS, IS std baseline, IS std hybrid, n_samples)
RESULTS = [
    ("MNIST",    54.4459, 46.8944, 1.9238, 1.9252, 0.0178, 0.0164, 10_000),
    ("CelebA",   16.0085, 14.6682, 2.3222, 2.2537, 0.0494, 0.0724,  7_500),
    ("CIFAR-10", 32.8998, 27.6049, 8.7243, 8.8674, 0.2801, 0.1931, 10_000),
]


def _apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       10,
        "axes.linewidth":  0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon":  False,
    })


def main() -> None:
    _apply_paper_style()

    datasets = [r[0] for r in RESULTS]
    fid_base, fid_hyb = [r[1] for r in RESULTS], [r[2] for r in RESULTS]
    is_base, is_hyb = [r[3] for r in RESULTS], [r[4] for r in RESULTS]
    is_base_std, is_hyb_std = [r[5] for r in RESULTS], [r[6] for r in RESULTS]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=300)
    x = np.arange(len(datasets))
    width = 0.36

    # ---------- Left: FID (lower is better) ----------
    bars1 = axL.bar(x - width/2, fid_base, width, label="Annealed Langevin (baseline)",
                    color="#888888", edgecolor="black", linewidth=0.4)
    bars2 = axL.bar(x + width/2, fid_hyb, width, label=r"Annealed $\alpha$-Half-Denoising (ours)",
                    color="#1f77b4", edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars1, fid_base):
        axL.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width()/2, v),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8)
    for bar, v in zip(bars2, fid_hyb):
        axL.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width()/2, v),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8, fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels(datasets)
    axL.set_ylabel(r"FID $(\downarrow)$")
    axL.set_title("FID: ours is lower across all 3 datasets")
    axL.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axL.legend(loc="upper right", fontsize=8.5)

    # ---------- Right: IS (higher is better, with std) ----------
    axR.bar(x - width/2, is_base, width, yerr=is_base_std, capsize=3,
            label="Annealed Langevin (baseline)",
            color="#888888", edgecolor="black", linewidth=0.4,
            error_kw=dict(ecolor="black", lw=0.7))
    axR.bar(x + width/2, is_hyb, width, yerr=is_hyb_std, capsize=3,
            label=r"Annealed $\alpha$-Half-Denoising (ours)",
            color="#1f77b4", edgecolor="black", linewidth=0.4,
            error_kw=dict(ecolor="black", lw=0.7))
    axR.set_xticks(x); axR.set_xticklabels(datasets)
    axR.set_ylabel(r"Inception Score $(\uparrow)$")
    axR.set_title("IS: comparable or slightly better")
    axR.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle("Annealed Langevin vs $\\alpha$-Half-Denoising on 3 image datasets",
                 fontsize=11)
    fig.tight_layout()

    out = Path("assets/results.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
