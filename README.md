<div align="center">

# marionette

**A polygraph for LLMs.** Chat with an open model while a linear probe on its activations
flags lies as they are written, a sparse autoencoder shows the concepts lighting up inside it,
and a truth-serum slider steers it back toward honesty.

[![ci](https://github.com/aamodbhatt/marionette/actions/workflows/ci.yml/badge.svg)](https://github.com/aamodbhatt/marionette/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg)
![model: Qwen3-4B](https://img.shields.io/badge/model-Qwen3--4B-6f42c1.svg)

![marionette catching a lie, then the truth serum](docs/assets/demo.gif)

<sub>Recorded from the running app (Qwen3-4B, greedy decoding). Playing a car salesman, the model
denies a known transmission fault and the needle reads 1.00. With the truth serum at alpha 0.5
it admits the fault and the needle drops to 0.24. The preset and strength were chosen by a rule
fixed before recording (<code>scripts/capture_demo.py</code>, log in
<code>docs/assets/demo.json</code>); the first preset, from a training family, met it.</sub>

</div>

## Why

Language models can say things they "know" are false. Asking them is circular. Their
activations are not: if lying leaves a consistent trace in the residual stream, a linear
probe can read it while the model writes, without trusting a word it says. marionette turns
that idea into an instrument you can watch, and measures how far it really generalises.

## What it does

- **Lie detector.** A logistic probe on one layer's residual stream scores every generated
  token. The needle is the calibrated verdict on the reply so far; the underline is each token's
  share. The flag threshold is set for a 1% false-positive rate on benign chat.
- **Thought stream.** A BatchTopK sparse autoencoder (16k latents) at the same layer; the top
  active auto-labelled features scroll by per token.
- **Truth serum.** Subtracts the "lying minus honest" activation direction during generation,
  with its effect on deception and on capability measured, plus sliders for individual SAE
  features.
- **Research dashboard.** AUROC by layer, the cross-dataset generalisation matrix, the
  steering dose-response curve and SAE statistics, all from committed result files.
- **Deception datasets,** built by code in this repo: true/false facts, instructed lying,
  incentive-driven role-play scenarios in 18 families (disjoint train/test families), and
  controls (confessions, fiction about con artists, playful personas, benign chat).

<p align="center">
  <img src="docs/assets/polygraph.png" width="49%" alt="polygraph tab: the car-salesman preset denies the fault, needle at 1.00, deception detected">
  <img src="docs/assets/truth_serum.png" width="49%" alt="the same turn with the truth serum at alpha 0.5: the model admits the fault, needle at 0.24">
</p>

## Headline results

Qwen3-4B (bf16, non-thinking mode), probe at layer 19 of 36. Datasets, activations and probes
ran on a single NVIDIA A40 (48 GB); the truth-serum evaluation, SAE, feature labels and demo on
a single NVIDIA RTX A6000 (48 GB). Both GPUs were shared with other jobs; 8 vCPU. Every number
below is in `results/qwen3-4b/*.json` or `weights/qwen3-4b/features.json`.

| What was measured (held-out data only) | Result |
|---|---|
| AUROC, **spontaneous lies** in 7 held-out families (the model chose to lie; 31 lies, 112 honest) | **0.896** |
| AUROC, scripted role-play lies in the same held-out families (121 lies, 147 honest) | 0.999 |
| Recall at the 1%-benign-FPR threshold: scripted / spontaneous lies (held-out) | 97.5% / 38.7% |
| False-positive rate actually measured on benign chat the threshold never saw (n = 120) | 3.3% |
| False positives on controls: confessions / fiction about deception / playful personas | 0% / 30% / 26% |
| Calibration error (ECE): scripted / spontaneous, held-out | 0.025 / 0.053 |
| Transfer to plain false statements ("The capital of Peru is Quito.") | 0.80 AUROC |
| Truth serum: deception rate of scripted liars, alpha = 0 vs 0.5 (96 held-out scenarios each) | 81% → 58% |
| Truth serum: deception rate when lying is the model's choice (spontaneous), alpha = 0 vs 0.5 | 31% → 34% (no reduction) |
| Capability (factual QA accuracy, 96 questions) at the same alpha | 97% → 95% |
| SAE (16k latents, layer 19): CE loss recovered / L0 / median per-token FVU | 0.964 / 43.5 / 0.36 |
| Auto-label detection AUROC, real vs shuffled labels (300 features) | 0.59 vs 0.50 |

<p align="center">
  <img src="docs/assets/auroc_by_layer.png" width="49%" alt="AUROC by layer">
  <img src="docs/assets/generalisation.png" width="49%" alt="generalisation matrix">
</p>
<p align="center">
  <img src="docs/assets/steering.png" width="49%" alt="truth serum dose-response">
  <img src="docs/assets/scores.png" width="49%" alt="score distributions">
</p>

<details>
<summary>The app's research tab, which draws the same result files</summary>

![research tab: AUROC by layer, generalisation matrix, truth-serum dose-response, SAE statistics and top features](docs/assets/research.png)

</details>

**What the numbers say.**

1. **Scripted lies are too easy.** A probe reaches AUROC 0.96 on held-out scripted role-play
   at *layer 0*, where the model cannot yet represent "I am lying". It can only be reading the
   script ("you have decided to hide this") through attention. That is why the layer was chosen
   on spontaneous lies, where every reply to a given prompt is equally plausible a priori.
2. **Chosen lies live mid-network.** On spontaneous lies the probe climbs from about 0.62
   (layers 0 to 5) to a plateau of 0.88 to 0.92 at layers 17 to 23, then decays toward the
   output. Layer 19 was picked on the selection set and scores 0.896 on held-out families.
3. **Training data matters more than the probe.** A probe trained only on true/false facts
   transfers to instructed lies (1.00) but not to spontaneous ones (0.46): "false statement"
   and "deceiving someone" are different directions. Training on role-play transfers best (0.91).
4. **Fiction is the weak spot.** At the 1%-benign threshold the probe flags 30% of stories
   about con artists and 26% of playful in-character replies, but no confessions. It responds
   to saying untrue things in character, not just to deceiving the listener.
5. **The truth serum is a blunt instrument.** Subtracting the lie-minus-honest direction at
   layers 19 to 22 cuts the scripted liars' deception rate from 81% to 58% at alpha 0.5 while
   QA accuracy barely moves (97% to 95%). It does nothing for lies the model chose: the
   spontaneous rate goes 31%, 34%, 47% at alpha 0, 0.5 and 1 (n = 96 each). Past alpha 1 the
   model breaks: from 1.5 up it writes empty replies or rows of dots, the judge rates 89% to
   100% of replies unclear, and QA accuracy is 0% to 1%. The "zero deception" at high alpha is
   collapse, not honesty. alpha 0.5 is the largest tested strength that keeps QA accuracy
   within 10 points of the unsteered model.
6. **The SAE explains the probe only in part.** Splicing it in costs 0.40 nats of CE loss
   (96.4% recovered against zero-ablation), but the median token keeps 36% of its variance
   unexplained: the global FVU of 0.009 is flattered by the top 1% highest-norm tokens, which
   hold 98% of the variance. The probe's lie signal is spread thin. The 10 latents that push its
   logit hardest toward "lie" carry only 37% of the positive contribution of the 300 latents
   studied, and no labelled latent separates lies from honest replies as well as the probe
   (best activation AUROC 0.88 on the training replies). The labels are weak too: given its
   own auto-labels, Qwen3-4B tells activating snippets from random ones with mean AUROC 0.59,
   against 0.50 with shuffled labels (0.66 vs 0.51 for the 50 latents the probe leans on most).
   The top contributor is labelled "call to action"; one latent is labelled "lie detection".
   Treat the labels as hypotheses.

## Quickstart

The probe weights for Qwen3-4B are in `weights/qwen3-4b/` (a few hundred KB). The model is
downloaded from Hugging Face on first run (about 8 GB). A GPU with 12 GB or more is
recommended; CPU works but is slow.

**Windows (PowerShell)**

```powershell
git clone https://github.com/aamodbhatt/marionette; cd marionette
uv sync
uv run marionette serve -c configs/qwen3-4b.toml   # open http://127.0.0.1:8096
```

**macOS / Linux**

```bash
git clone https://github.com/aamodbhatt/marionette && cd marionette
uv sync
uv run marionette serve -c configs/qwen3-4b.toml   # open http://127.0.0.1:8096
```

The SAE (about 340 MB) is not in git. Without it the app runs with the thought stream switched
off; train it with `marionette sae tokens` and `marionette sae train` (about 9 GPU-minutes for
7.2M tokens on a shared RTX A6000), then run `marionette features` (about 5 GPU-minutes). The
committed `weights/qwen3-4b/features.json` labels the latents of the SAE trained for this
study, and a retrained SAE numbers its latents differently, so the labels must be rebuilt.

**Laptop profile.** `configs/laptop.toml` targets Qwen3-1.7B, which runs on a 6 GB GPU. Its
probes must be trained on its own activations (`data`, `capture`, `probes` on a GPU machine).
Its probes were **not** trained in this run (the GPU budget went to the 4B study), so
`weights/qwen3-1.7b/` does not exist yet and `serve -c configs/laptop.toml` will tell you to run
`marionette probes` first. Expect a much weaker detector: probe quality falls with model size.

**Reproduce everything** (one GPU slot, about 30 GPU-minutes on a shared 48 GB A40 or RTX A6000):

```bash
uv run marionette sae tokens -c configs/qwen3-4b.toml   # CPU + network: tokens to uint32
bash scripts/pipeline.sh configs/qwen3-4b.toml          # data, capture, probes, steer-eval, SAE, features
uv run python scripts/make_plots.py                     # figures in docs/assets
```

## How it works

```mermaid
flowchart LR
  subgraph data["datasets (generated)"]
    F[facts<br/>true/false] --- I[instructed<br/>lie vs honest] --- R[role-play<br/>18 families] --- C[controls<br/>fiction, confessions,<br/>personas, benign]
  end
  data --> G[Qwen3-4B<br/>on-policy replies]
  G --> J{verify<br/>answer keys +<br/>Qwen3-8B judge}
  J --> A[pooled activations<br/>every layer]
  A --> P[probes<br/>layer sweep on<br/>validation families]
  P --> W[(probe +<br/>truth-serum vectors)]
  T[FineWeb-Edu +<br/>UltraChat tokens] --> S[BatchTopK SAE<br/>streamed, 16k latents]
  S --> L[auto-labels +<br/>detection scores]
  W --> E[LiveEngine]
  S --> E
  L --> E
  E -->|NDJSON stream| UI[browser UI<br/>needle, heat, thoughts, sliders]
```

**The probe.** For each reply we average the residual stream over the reply's tokens at every
layer, and fit logistic-regression and difference-of-means probes on instructed lies plus
role-play from the *training* families. The layer is picked on *validation* families; the
reported numbers come from *test* families the probe never saw: insider trading, food
safety, software releases, a vault password, lab results, supplements and charity funds.

**Why the per-token heat adds up.** A logistic probe is an affine map, so its logit on the
mean activation equals the mean of its per-token logits. The live needle is the calibrated
running mean of the per-token scores: exactly the probe's verdict on the reply so far.

**Why the needle can't cheat.** Steering edits the residual stream at the probe layer and the
three after it, but the probe reads that layer *before* the edit. If the needle drops under
the truth serum, it is because the model wrote different words.

**Spontaneous lies.** In the scripted conditions the system prompt says whether the character
lies, so a probe could read the instruction instead of the reply. The *spontaneous* set removes
that: identical prompts give the character stakes but no script, the model decides, and a judge
labels what it did.

**Labels you can check.** Instructed answers are verified against answer keys. Role-play replies
must be judged (by Qwen3-8B, loaded after the generator is freed) to match their condition, or
they are dropped and counted. The judge agrees with the answer keys on 96% (300 of 312) of instructed
replies.

**The SAE.** BatchTopK (k = 64 on average) with an AuxK loss for dead latents, trained on 7.2M
tokens of FineWeb-Edu and UltraChat streamed through the truncated model. Activations never
touch disk. Because the probe is linear, the reconstruction's probe logit splits exactly into
per-latent terms, which ranks the features the probe leans on. The 120 latents that push the
probe hardest toward "lie", plus the 180 that fire most often in the dialogs, are labelled by
Qwen3-4B itself from their top activations, and each label is scored by asking the model to
tell held-out activating snippets from random ones, against a shuffled-label baseline.

## Limitations (read these)

- **Probes generalise less than their in-distribution AUROC suggests.** A linear probe can
  break under distribution shift (new formats, longer agentic transcripts, a model trained to
  obfuscate against it), and this study already shows the gap: 0.999 on scripted lies versus
  0.896 on spontaneous ones, and 30% false positives on fiction about deception. Held-out
  families here are still short, single-turn, templated role-play.
- **Scripted versus chosen lies.** Most positives are lies the model was told to tell. The
  spontaneous test set is small (143 replies, 31 of them lies), so its AUROC has wide error bars.
- **The judge is a model.** Qwen3-8B labels role-play replies and steered replies. It agrees with
  answer keys on 96% of instructed QA, but role-play is harder to judge than arithmetic.
- **Small study.** 2,471 labelled replies, one 4B model, one seed, and 7.2M SAE
  tokens: a fraction of what published SAEs use. Treat feature labels as hypotheses.
- **Not a product.** A probe that can be read can be trained against. Do not use this to make
  decisions about people.

## Design decisions

See [docs/DESIGN.md](docs/DESIGN.md) for goals, invariants, the data model, failure modes and
alternatives. In short: the model is Qwen3-4B, because the 75-GPU-minute budget had to cover
an SAE as well; instructed lying uses a one-shot worked example, because Qwen3-4B in
non-thinking mode answers truthfully when merely told to lie; the judge is the 8B, because the
4B judge accepted "there's no password" as honest; the probe layer is the earliest within 0.01
validation AUROC of the best, leaving room downstream for the truth serum; and on a shared GPU
`scripts/pipeline.sh` is meant to run inside a single queue slot, so the wait happens once, not
per stage.

## Testing

`uv run pytest` runs 56 tests in a few seconds on CPU, with no network and no downloads. They
cover probe math (AUROC against a brute-force definition, the mean-pooling identity, FPR
thresholds and Platt calibration; property-based with Hypothesis); dataset label correctness
(arithmetic and table facts, disjoint families, minimal pairs, judge-verdict rules); SAE math
(exact BatchTopK sparsity, AuxK touching only dead latents, unit decoder norms, the parallel
gradient removed, round trips); hooks and steering on a tiny randomly initialised Qwen3 built
from a config (read-before-edit, capture equals `output_hidden_states`); the live engine; and
the HTTP API with a stub engine. CI runs ruff, `mypy --strict` and the tests on Ubuntu with
CPU torch.

## Project layout

```
src/marionette/
  config.py          typed TOML profiles
  lm.py              hookable model wrapper (reads before edits)
  data/              facts, instructed QA, role-play families, controls, judge, build
  capture.py         per-layer pooled response activations -> memmaps
  probes.py          logistic / diff-of-means probes, Platt, AUROC, FPR thresholds
  evaluation.py      layer sweep, generalisation matrix, calibration, export
  steering.py        probe bundle + truth serum      steer_eval.py  dose-response
  sae.py             BatchTopK + AuxK SAE            sae_train.py   tokens, training, eval
  features.py        deception alignment, top activations, auto-labels, detection scoring
  server/            FastAPI app, LiveEngine, presets, static UI (vanilla JS)
configs/             qwen3-4b (main), laptop (Qwen3-1.7B), smoke
weights/qwen3-4b/    committed probe, truth-serum vectors, feature labels
results/qwen3-4b/    committed result JSON behind every number in this README
scripts/             pipeline, demo capture, plots
```

## Dataset licences

| Data | Source | Licence |
|---|---|---|
| Facts, instructed QA, role-play scenarios, confessions, fiction and persona prompts | written for this repo (`src/marionette/data/`) | MIT |
| Benign chat prompts; SAE chat text | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT |
| SAE web text | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample/10BT`) | ODC-By 1.0 |
| Models | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (judge) | Apache-2.0 |

Generated replies are model outputs from the Apache-2.0 Qwen3 models. The datasets are rebuilt
by `marionette data` and `marionette sae tokens`, not shipped. The one exception is
`weights/qwen3-4b/features.json`, which keeps 900 short top-activation excerpts (three per
labelled latent) from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(ODC-By 1.0), [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (MIT)
and this repo's own dialogs.

## Roadmap

- Multi-turn and agentic transcripts (where probes are known to struggle).
- Train the probe adversarially against an obfuscating fine-tune.
- Publish the SAE weights on the Hugging Face Hub so the thought stream works out of the box.
- Laptop-profile probes for Qwen3-1.7B, trained on its own activations.

## License

MIT, Copyright (c) 2026 Aamod Bhatt.
