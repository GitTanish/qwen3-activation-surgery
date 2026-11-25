# Qwen3 Activation Surgery

Experiments in **activation steering** on Qwen3-0.6B. Part 1A extracts a
Difference-of-Means concept direction from the residual stream; Part 1B injects
it with a forward hook and sweeps strengths.

This repo contains the extraction and inference-steering scripts only. The
abliteration (`P = I - r r^T`) and single-neuron probe scripts are local-only and
gitignored.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

The model id is hardcoded per script (`MODEL_ID = "Qwen/Qwen3-0.6B"`); edit it to
use a larger Qwen3 checkpoint. Weights download automatically on first run.

## Files

| File | Role |
|------|------|
| `prompts.json` | Two concept sets, 12 pairs each (`mood`, `refusal`) |
| `1_extract_vector.py` | Difference-of-Means -> `artifacts/mood_steering_matrix.pt` |
| `2_steer_inference.py` | Forward-hook injection + strength sweep |
| `artifacts/mood_steering_matrix.pt` | Steering tensor, shape `(28, 1024)` for Qwen3-0.6B |
| `artifacts/steering_sweep.txt` | Recorded generations for strengths -2...+2 |

## Prompt sets

| Concept | `positive_label` | `negative_label` | `+alpha` drives toward |
|---------|------------------|------------------|-------------------------|
| `mood` | sad | happy | sad |
| `refusal` | refusal | compliance | refusal |

`mood` is wired into the committed scripts. `refusal` is included for the local
abliteration script.

## Math cheat sheet

Let `h_l(x)` be the mean-pooled residual stream at layer `l`.

**1. Difference of Means (Part 1A)**

```
r_l = E_{P+}[h_l(x)] - E_{P-}[h_l(x)]
```

For `mood`, `P+` = sad prompts and `P-` = happy prompts, so `r_l` points toward
sad. `artifacts/mood_steering_matrix.pt` stores `[num_layers, hidden_size]`.

**2. Inference injection (Part 1B)**

```
r_hat = r_l / ||r_l||
h_l <- h_l + alpha * r_hat
```

A forward hook applies this at every forward pass, so it fires on every generated
token. `alpha` is the strength; `TARGET_LAYER` chooses `l`.

**3. Abliteration (local-only, not in this repo)**

With `r` unit-normalized, `P = I - r r^T`. For a weight `W` writing into the
residual stream, `W' = P W` makes `W' x . r = 0` for all `x`.

**4. Neuron probing (local-only, not in this repo)**

For MLP intermediate unit `j`, discovery score `s_j = |E_{P+}[a_j] - E_{P-}[a_j]|`;
clamping pins `a_j = c` during generation.

## Usage

```bash
uv run python 1_extract_vector.py
uv run python 2_steer_inference.py
```

Both scripts are configured by constants at the top of the file:

- `1_extract_vector.py`: `CONCEPT = "mood"`, writes `artifacts/{CONCEPT}_steering_matrix.pt`.
- `2_steer_inference.py`: `TARGET_LAYER = 20`, strengths `[-2, -1, 0, 1, 2]`,
  80 new tokens per generation.

Edit those constants to sweep other layers or strengths.

## Results so far

Step 1:

```
Model: Qwen/Qwen3-0.6B | Saved: artifacts\mood_steering_matrix.pt | shape: (28, 1024)
```

Step 2 (`artifacts/steering_sweep.txt`), neutral prompt
"Tell me a short story about a cat and a dog.", layer 20:

| strength | observation |
|----------|-------------|
| -2 | neutral adventure story |
| -1 | neutral adventure story |
| 0 | baseline |
| +1 | neutral adventure story |
| +2 | neutral adventure story, ends on "happy/proud" |

At layer 20 with unit-normalized vectors, +/-2 did not produce an obvious mood
shift on this prompt. Raise the strength, try another layer, or extract from
model completions instead of prompts.

## Caveats

- Only one neutral prompt and one layer were tested; there is no statistical claim.
- The `refusal` prompts are verbalized refusals/compliance, not real harmful requests.
- The artifact is model-specific: `(28, 1024)` only matches Qwen3-0.6B.
- `torch_dtype=` is deprecated; the warning is harmless (use `dtype=` to silence).
- Tested on CPU (bfloat16); the CUDA float16 path is untested.
