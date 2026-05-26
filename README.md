# Annealed α-Half-Denoising for score-based generative sampling

**MVA (Probabilistic Graphical Models) course project — Faissal Izermine & Dhia Garbaya.**

We study the **sampling step** of NCSN-style score-based models. The
baseline — *Annealed Langevin Dynamics* (Song & Ermon, NeurIPS 2019) —
explores the data manifold well but produces samples with a residual noise
bias because the score is learned on noisy data. Hyvärinen (2025) recently
proposed *Half-Denoising* to remove that bias at a fixed noise level. We
combine the two and obtain **consistently lower FID across MNIST, CelebA
and CIFAR-10** while keeping the exploration benefits of annealing.

> Full method, derivation and experiments: [`REPORT.pdf`](REPORT.pdf).

![FID / IS comparison](assets/results.png)

---

## 1. Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended (sampling 10k CIFAR-10 / CelebA images via Langevin
dynamics takes hours on a single GPU).

### Pretrained NCSN checkpoints

We reuse the pretrained NCSN checkpoints from the original Song & Ermon
release. Download and extract them to the repo root:

```bash
gdown https://drive.google.com/uc?id=1BF2mwFv5IRCGaQbEWTbLlAOWEkNzMe5O
unzip run.zip
```

The training code for these checkpoints is included (`models/`,
`losses/`, `runners/*.py`) and is largely unchanged from upstream.

## 2. Method in one paragraph

Score-based models learn `s_θ(x̃, σ) ≈ ∇_{x̃} log q_σ(x̃)`, the score of
the **noisy** distribution at scale σ. To draw clean samples, the baseline
runs Langevin dynamics on a decreasing schedule `{σ_1 > σ_2 > … > σ_L}`,
using step size `α_i = ϵ · σ_i² / σ_L²`. Hyvärinen's noise-corrected
update shows that at the limit `μ = σ² / 2`, the right-hand noise term
vanishes and the recurrence collapses to a **deterministic half-denoising
step** `x_{t+1} = x̃_t + (σ²/2) Ψ_{x̃}(x̃_t)` — which targets the *clean*
distribution. Our method, **Annealed α-Half-Denoising**, applies the
half-denoising update but scales it by the annealed step `α_i` to keep
transitions smooth across noise levels. See §4 of the report and
[`runners/anneal_runner.py`](runners/anneal_runner.py) (`half_denoising_*`
methods) for the implementation.

## 3. Running the sampler

The single entry point is `main.py`. The key flag added by this fork is
`--sampling_type`, which selects between the upstream `ordinary` (Annealed
Langevin) and our `half_denoising` (Annealed α-Half-Denoising).

### Baseline (Annealed Langevin)

```bash
python main.py \
    --runner AnnealRunner \
    --heavy_test \
    --doc cifar10 \
    --sampling_type ordinary \
    --n_samples 10000 \
    -o samples/anneal
```

### Ours (Annealed α-Half-Denoising)

```bash
python main.py \
    --runner AnnealRunner \
    --heavy_test \
    --doc cifar10 \
    --sampling_type half_denoising \
    --n_samples 10000 \
    -o samples/half_denoising
```

Replace `--doc cifar10` with `mnist` or `celeba` to switch datasets;
each `--doc` value corresponds to a config under [`configs/`](configs/).

Both commands load the **same** pretrained model and differ only in the
sampling procedure — so any difference in the output is attributable to
the update rule.

## 4. Evaluation

Quantitative metrics are computed with
[`torch-fidelity`](https://github.com/toshas/torch-fidelity) from
[`evaluation_colab.ipynb`](evaluation_colab.ipynb) (kept as a Colab
notebook because of its lightweight GPU requirements). Workflow:

1. Run the sampler above to fill `samples/<run>/`.
2. Open the notebook and point it to the sample folder.
3. It computes Inception Score and FID against the dataset.

The Kaggle notebook [`sampling_kaggle.ipynb`](sampling_kaggle.ipynb) is the
sampling driver we actually used (Kaggle gives 30 h/week of uninterrupted
GPU).

## 5. Results

All numbers from REPORT.pdf §6, averaged over the reported sample budget
(`n=10k` for MNIST and CIFAR-10, `n=7.5k` for CelebA).

| Dataset | Method | IS ↑ | FID ↓ |
|---|---|---:|---:|
| MNIST    | Annealed Langevin (baseline) | 1.9238 ± 0.018 | 54.45 |
| MNIST    | **Annealed α-Half (ours)**   | **1.9252 ± 0.016** | **46.89** |
| CelebA   | Annealed Langevin (baseline) | **2.3222 ± 0.049** | 16.01 |
| CelebA   | **Annealed α-Half (ours)**   | 2.2537 ± 0.072 | **14.67** |
| CIFAR-10 | Annealed Langevin (baseline) | 8.7243 ± 0.280 | 32.90 |
| CIFAR-10 | **Annealed α-Half (ours)**   | **8.8674 ± 0.193** | **27.60** |

CIFAR-10 shows the strongest improvement (FID −16%). Inception scores
are within noise on all three datasets — the gain is in distribution
matching, not in per-image quality, which is the prediction made by the
half-denoising theory (§2.3 of the report).

To re-render the plot above from these numbers, run:

```bash
python make_results_plot.py
```

## 6. Qualitative comparison

Hybrid sampling (Algorithm 3 from the report) versus the unstable Algorithm 2:

![BEFORE / AFTER CelebA](assets/before_after_celeba.png)

Animated views of the sampling dynamics:

<table>
<tr>
  <th>1. Annealed Langevin (baseline)</th>
  <th>2. Annealed Half-Denoising (Alg. 2, unstable)</th>
  <th>3. Annealed α-Half-Denoising (Alg. 3, ours)</th>
</tr>
<tr>
  <td><img src="assets/celeba_small.gif"                     alt="celeba baseline"></td>
  <td><img src="assets/movie_half_denoising_celeba_high_step.gif" alt="celeba half-denoising"></td>
  <td><img src="assets/movie_half_denoising_alpha_step.gif"  alt="celeba alpha-half"></td>
</tr>
</table>

## 7. Repository structure

```
.
├── REPORT.pdf                    Full project report (method + experiments)
├── main.py                       Entry point: dispatch to a Runner
├── runners/
│   ├── anneal_runner.py          ← our half_denoising_* methods live here
│   ├── baseline_runner.py
│   ├── scorenet_runner.py
│   └── toy_runner.py
├── models/                       NCSN architectures (upstream Song & Ermon)
├── losses/                       Denoising / sliced score-matching losses
├── datasets/                     MNIST / CelebA / CIFAR-10 wrappers
├── configs/                      One YAML per dataset
├── evaluation_colab.ipynb        FID / IS via torch-fidelity
├── sampling_kaggle.ipynb         GPU sampling driver
├── make_results_plot.py          Re-renders assets/results.png from §5 numbers
└── assets/                       Plots + sampling GIFs
```

The user-side contribution (added on top of upstream NCSN) is concentrated
in the `half_denoising_*` methods of
[`runners/anneal_runner.py`](runners/anneal_runner.py) and the dispatch
logic that exposes them via the `--sampling_type` flag.

## 8. Acknowledgements

This codebase is forked from the **NCSN** reference implementation by
Song & Ermon ([ermongroup/ncsn](https://github.com/ermongroup/ncsn)).
The pretrained checkpoints are theirs. Our contribution is the
α-Half-Denoising sampler and the comparative evaluation.

## 9. Citation

```bibtex
@misc{izermine2026halfdenoising,
  title  = {Exploring Half-Denoising and Annealed Langevin Dynamics for Generative Sampling},
  author = {Izermine, Faissal and Garbaya, Dhia},
  year   = {2026},
  note   = {Master MVA, Probabilistic Graphical Models project.
            See \texttt{REPORT.pdf}.}
}
```

Underlying references:

- Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.
- Hyvärinen, A. (2025). *A noise-corrected Langevin algorithm and sampling by half-denoising*. arXiv:2410.05837.
- Beyler, E. & Bach, F. (2025). *Optimal Denoising in Score-Based Generative Models: The Role of Data Regularity*. arXiv:2503.12966.
