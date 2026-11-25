# Qwen3 Activation Surgery

Small, self-contained experiments in **activation steering** on Qwen3: find a
concept direction in the residual stream, inject it at inference, project it out
of the weights permanently, and poke individual MLP neurons.

## Setup

```bash
pip install -r requirements.txt
```

All scripts default to `Qwen/Qwen3-0.6B` so they run on modest hardware. Override
with `--model Qwen/Qwen3-8B` (or any local path) when you want the real thing.
Download happens automatically on first run.

## Files

| File | Role |
|------|------|
| `prompts.json` | Two concept sets, 12 contrasting pairs each |
| `1_extract_vector.py` | Difference-of-Means -> `steering_vec.pt` |
| `2_steer_inference.py` | Forward-hook injection + layer/strength sweep |
| `3_abliterate_weights.py` | Permanent weight edit `P = I - r r^T` |
| `4_probe_neurons.py` | Single MLP neuron discovery & clamping |

## Prompt sets

| Concept | `positive_label` | `negative_label` | Direction `+alpha` drives toward |
|---------|------------------|------------------|----------------------------------|
| `mood` | sad | happy | sad |
| `refusal` | refusal | compliance | refusal |

Each set has 12 pairs, so the mean is reasonably stable instead of being driven
by one exotic prompt. `positive` is always the *target* of the extracted vector.

## Math cheat sheet

Let `h_l(x)` be the residual-stream hidden state at layer `l`, mean-pooled over
the prompt tokens. For the `positive` prompts `P+` and `negative` prompts `P-`:

**1. Difference of Means (Part 1A)**

```
r_l = E_{x in P+}[h_l(x)] - E_{x in P-}[h_l(x)]
```

`steering_vec.pt` stores `{concept: [num_layers, hidden_size]}` — one direction
per layer. For `mood` this is sad - happy; for `refusal` it is refusal - compliance.

**2. Inference injection (Part 1B)**

```
h_l <- h_l + alpha * r_l
```

Applied as a forward hook on layer `l`, so it fires on every forward pass —
including every token during generation. `alpha` (strength) and `l` are the two
knobs swept in `2_steer_inference.py`.

**3. Abliteration / projection (Part 1C)**

With `r` unit-normalized, `P = I - r r^T` is the projector onto the subspace
orthogonal to `r`. For a weight `W` whose output lives in the same space as `r`:

```
W' = P W   =>   W' x · r = 0   for all x
```

Applied to `down_proj` (MLP output) and `o_proj` (attention output) at every
layer, so the model can no longer *write* the concept direction into the
residual stream. This is a permanent edit saved as a new checkpoint.

**4. Neuron probing (Part 2)**

A neuron is one intermediate unit feeding `down_proj`, with activation `a_j`.
Discovery score:

```
s_j = |E_{P+}[a_j] - E_{P-}[a_j]|
```

Clamping sets `a_j = c` during generation to isolate that neuron's behaviour.

## Usage

```bash
# 1A: build the vectors (run this first)
python 1_extract_vector.py

# 1B: sweep layers/strengths on a mood probe (happy prompts by default)
python 2_steer_inference.py --concept mood --layers 6,12,18,24 --strengths 0,1,2,4

# 1C: permanently remove the refusal direction, save a checkpoint
python 3_abliterate_weights.py --concept refusal --output abliterated_model

# 2: find the most mood-sensitive neurons at a layer and clamp them
python 4_probe_neurons.py --concept mood --layer 12 --top-k 5
```

Tip: start with `--strengths 0,2,4` — `0` is your baseline, and large values
usually degrade fluency. For Qwen3-0.6B, valid layer indices are `0..27`.

## Experiment outputs

> Template only — replace with your own results after running the scripts.

### Part 1A — extracted vectors

```
mood:      [28, 1024] over 12 pairs (sad - happy)
refusal:   [28, 1024] over 12 pairs (refusal - compliance)
Saved steering_vec.pt
```

### Part 1B — steering sweep (illustrative)

| layer | strength | observed effect |
|-------|----------|-----------------|
| 12    | 0        | baseline tone   |
| 12    | 2        | noticeable shift toward sad |
| 12    | 4        | strong sad shift, some repetition/degradation |

### Part 1C — abliteration

```
layer 0..27: projected out direction
Saved edited model to abliterated_model
```

Before/after: compare refusal behaviour on `refusal` prompts using
`python 2_steer_inference.py --model abliterated_model --concept refusal --strengths 0`.

### Part 2 — neuron probe

```
layer 12: top 5 neurons (index, score)
  431: 0.92
  887: 0.76
  ...
```

## Notes / caveats

- The `refusal` set contains *verbalized* refusal/compliance sentences, not
  actual harmful requests, so the extracted direction may capture refusal
  wording more than the model's refusal computation. Validate on real prompts.
- Directions are extracted from prompts only; if you care about *generated*
  style, extract from model completions instead.
- `P = I - r r^T` assumes `r` is normalized; the script normalizes it.
- Projection is tangential: it removes the direction at write time but does not
  retrain the model. Expect some collateral damage at high-coupling layers.
- Neuron indices are per-layer, in `down_proj`'s **input** (intermediate) space.

## Skipped / not checked

- No chat-template handling: prompts are passed as raw text, so results reflect
  base-model behaviour. Add `tok.apply_chat_template(...)` if you want chat-style
  inputs.
- No GPU was available here, so the scripts were written but not executed against
  a real checkpoint; expect to tune defaults (layer, strength) for your model.
