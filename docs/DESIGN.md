# marionette: design

## Goals

1. **Detect deception from activations, live.** A linear probe reads the residual stream of an
   open chat model while it writes, and shows a calibrated verdict per token and per reply.
2. **Explain the detector.** A sparse autoencoder at the probe layer decomposes the same
   activations into labelled features, so we can say which concepts the probe leans on.
3. **Reduce deception, measurably.** A "truth serum" steering vector built from the same data
   is subtracted during generation; its effect on lying and on capability is measured.
4. **Be honest about the evidence.** Generalisation is measured on scenario *families* the
   probe never saw, false positives on benign chat the threshold never saw, and every number in
   the README comes from a committed result file.

## Non-goals

- A production lie detector. Probes this small and data this narrow are research tooling.
- Jailbreaks, refusal removal or any safety bypass. Steering only ever subtracts the
  deception direction or nudges labelled SAE features within a bounded range.
- State-of-the-art SAE quality. The SAE exists to interpret the probe, at a 75-GPU-minute budget.

## Pipeline and data model

```
prompts (templates, fact tables, UltraChat)  ->  Example{messages, label=None|0|1}
  -> generate (studied model, on-policy)      ->  Example.response
  -> verify (answer keys; LLM judge)          ->  Example.label, kept, drop_reason
  -> capture (hooks on every block)           ->  acts/mean.npy, acts/max.npy  [n, layers, d]
  -> probes (sweep, select on val families)   ->  weights/<run>/probe.{safetensors,json}
  -> steer-eval (generate under steering, judge unsteered) -> results/<run>/steering.json
  -> sae tokens / sae train / sae eval        ->  artifacts/<run>/sae/, results/<run>/sae.json
  -> features (alignment, labels, scoring)    ->  weights/<run>/features.json
  -> serve (LiveEngine + FastAPI + static UI)
```

Every stage reads the same TOML profile (`configs/*.toml`, validated by pydantic with unknown
keys rejected) and writes atomically (temp file plus rename), so a crashed stage never leaves a
file a later stage would mistake for a finished one.

`Example` (in `marionette/data/schema.py`) is the only record type: id, dataset, family, split,
condition, label, messages, response, ground truth, answer key, judge probabilities, and
whether it was kept (with the reason when not).

## Invariants

- **Disjoint families.** Train, validation and test role-play families never overlap
  (`test_family_splits_are_disjoint...`). The layer and L2 are chosen, and Platt calibration is
  fitted, on spontaneous replies from non-test families (never trained on); test families and
  the benign test half are scored once, by the final probe.
- **Minimal pairs.** Deceptive and honest role-play instances share the character, the hidden
  fact, the incentive and the question; only the decision sentence differs (tested).
- **Labels by construction, then verified.** Instructed answers must match (or avoid) the answer
  key; role-play replies must be judged to match their condition with confidence >= 0.7, else
  they are dropped and counted in `dataset.json`.
- **Read before edit.** Inside a block hook, reads happen before edits. The polygraph therefore
  never sees a residual the truth serum has already changed at that layer, and steering starts
  at the probe layer, so the needle can only drop when the *words* change.
- **Per-token heat decomposes the verdict exactly.** Probes are affine maps in raw residual
  space, so the probe's logit on a mean-pooled reply equals the mean of its per-token logits
  (tested). The live needle is the calibrated running mean; the underline is the per-token term.
- **Activations never touch disk** during SAE training; only uint32 token ids do.

## Key choices and alternatives

| Choice | Why | Alternative considered |
|---|---|---|
| Qwen3-4B, non-thinking mode | Fits the 75-GPU-minute budget with room for an SAE; thinking traces would make "response tokens" ambiguous. | Qwen3-8B: better behaved but ~2x the cost of every stage. |
| Qwen3-8B as the judge, loaded after the generator is freed | The 4B judge called "there's no password" honest; 8B is markedly better. Never co-resident, so each job stays under 22 GB. | Self-judging (cheaper, measurably worse). |
| One-shot demonstration for instructed lying | Qwen3-4B in non-thinking mode answers correctly when told to lie, whether the instruction is in the system prompt or the user turn (about 90% of "lies" were true in smoke runs). One worked example with a false (or, for the honest twin, true) answer reduces this but does not fix it: in the final run 165 of 240 lie attempts (69%) still contained the correct answer. Those are dropped and counted in `results/qwen3-4b/dataset.json`. | Instruction only, in either position. |
| Earliest layer within 0.01 selection-set AUROC of the best | Near-ties go to the layer with the most downstream room for the truth serum (which must start at the probe layer). | Pure argmax (picked the last layer in the smoke run, where steering has no effect). |
| Mean pooling for the shipped probe | Makes per-token heat an exact decomposition of the verdict. Max pooling is swept and reported. | Max pooling or per-token training. |
| Select the layer and calibrate on spontaneous lies from non-test families | Scripted role-play reaches AUROC 0.96 at layer 0, so it measures reading the script, not detecting a lie; fitting Platt scaling on perfectly separable scripted data gave a near-step, overconfident calibration (slope 9.3, ECE 0.148 on held-out spontaneous lies, versus 0.053 now). | Validation role-play families (tried first; picked layer 6). |
| Threshold at 1% FPR on half of the benign set, FPR reported on the other half | The headline FPR is measured on prompts the threshold never saw. | Threshold on the same prompts it is reported on (optimistic). |
| Truth serum = class-mean difference per layer, over a 4-layer band | Mean differences steer better than logistic weights, whose direction is shaped by variance. | Projection ablation; a single layer. |
| BatchTopK + AuxK SAE | Exact average L0, few dead latents, batch-independent threshold at inference. | ReLU + L1 (shrinkage), TopK (fixed per-token L0). |

## Failure modes and how they surface

- **The model refuses to follow a condition** (for example it confesses when scripted to lie):
  the example is dropped, and drop counts per reason are in `results/<run>/dataset.json`.
- **The judge is wrong:** measured on instructed QA, where the answer key is ground truth
  (`judge_vs_answer_key.accuracy`).
- **The probe keys on the system prompt rather than the reply:** controlled by the spontaneous
  set (identical prompts, the model chooses whether to lie) and by the fiction and confession
  controls (deception as subject matter, but honest).
- **Steering breaks the model instead of making it honest:** the judge's "unclear" rate and the
  QA accuracy are reported at every alpha.
- **No SAE on disk:** the server starts without the thought stream, and the UI says so.
- **Bad request:** pydantic validation returns 422. Engine errors (unknown feature id, feature
  steering without an SAE) arrive as an `error` event in the NDJSON stream.
- **GPU contention:** `scripts/pipeline.sh` chains every GPU stage of a profile so that, on a
  shared GPU, it can run inside a single queue slot and the wait happens once, not per stage.
  For this study the stages were run individually; `scripts/run_all.sh` (smoke run, main run,
  demo) has not been run end to end.
