import json
from contextlib import contextmanager
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
VECTOR_PATH = Path("artifacts/mood_steering_matrix.pt")
TARGET_LAYER = 21  # Peak semantic layer from Eval 1
STRENGTHS = [-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0]


@contextmanager
def steer_residual(model, layer_idx: int, direction: torch.Tensor, strength: float):
    unit_vec = (direction / (direction.norm() + 1e-8)).to(
        device=model.device, dtype=model.dtype
    )

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden + strength * unit_vec
        return (steered,) + output[1:] if isinstance(output, tuple) else steered

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def single_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        toks = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return ids


@torch.no_grad()
def get_next_token_logprobs(model, inputs: dict) -> torch.Tensor:
    logits = model(**inputs).logits[0, -1, :].to(torch.float32)
    return F.log_softmax(logits, dim=-1)


def main():
    steering = torch.load(VECTOR_PATH, map_location="cpu").to(torch.float32)
    layer_vec = steering[TARGET_LAYER]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()

    # Neutral prefix forcing an immediate emotional adjective/verb on the very next token
    messages = [{"role": "user", "content": "Complete this sentence in one line: Looking out the window, the entire street felt"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) + "Looking out the window, the entire street felt"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    sad_ids = single_token_ids(tokenizer, ["empty", "cold", "dark", "lonely", "quiet", "sad", "bleak"])
    happy_ids = single_token_ids(tokenizer, ["alive", "warm", "bright", "vibrant", "cheerful", "happy", "lively"])

    base_logprobs = get_next_token_logprobs(model, inputs)
    base_probs = base_logprobs.exp()

    results = []
    print(f"=== Eval 2: Layer {TARGET_LAYER} Dose-Response Curve ===")
    print(f"{'Strength':>8} | {'Sad-Happy ΔLogP':>16} | {'KL Damage':>10} | {'Top-1 Next Token'}")
    print("-" * 62)

    for s in STRENGTHS:
        if s == 0.0:
            lp = base_logprobs
        else:
            with steer_residual(model, TARGET_LAYER, layer_vec, s):
                lp = get_next_token_logprobs(model, inputs)

        # Sentiment margin: mean log-prob of sad markers minus happy markers
        margin = float(lp[sad_ids].mean() - lp[happy_ids].mean())
        # KL(P_base || P_steered) measures distribution distortion from unsteered baseline
        kl = float(F.kl_div(lp, base_probs, reduction="sum", log_target=False))
        top_tok = repr(tokenizer.decode([int(lp.argmax())]))

        print(f"{s:+8.1f} | {margin:+16.3f} | {kl:10.4f} | {top_tok}")
        results.append({"strength": s, "sad_minus_happy_logp": round(margin, 4), "kl_div": round(kl, 4), "top_token": top_tok})

    Path("results/eval_2_steering.json").write_text(
        json.dumps({"model": MODEL_ID, "layer": TARGET_LAYER, "sweep": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
