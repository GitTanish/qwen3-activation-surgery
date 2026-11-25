from contextlib import contextmanager
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
VECTOR_PATH = Path("artifacts/mood_steering_matrix.pt")
TARGET_LAYER = 20 # try one of the top layers printed by 1_extract_vector.py

@contextmanager
def steer_residual(model, layer_idx: int, direction: torch.Tensor, strength: float):
    """Add a normalized steering vector to layer_idx's output during forward passes."""
    unit_vec = (direction/ (direction.norm() + 1e-8)).to(model.device, dtype=model.dtype)

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden + strength * unit_vec
        return (steered,) + output[1:] if isinstance(output, tuple) else steered

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

@torch.no_grad()
def generate_reply(model, tokenizer, prompt: str) -> str:
    messages = [{"role":"user","content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt = True, enable_thinking = False
    )
    inputs = tokenizer(formatted, return_tensors = "pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature = 0.7,
        top_p = 0.8,
    )
    # slice off prompt tokens so only newly generated text is decoded
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map = "auto"
    )
    model.eval()

    steering_matrix = torch.load(VECTOR_PATH, map_location = "cpu")
    layer_vec = steering_matrix[TARGET_LAYER]

    neutral_prompt = "Tell me a short story about a cat and a dog."

    # sweep from -2.0 (opposite direction: happy) -> 0.0 (baseline) -> 2.0 (opposite direction: sad)
    for strength in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        print(f"\n-- Layer {TARGET_LAYER} | Strength: {strength:+.1F} ---")
        if strength == 0.0:
            print(generate_reply(model, tokenizer, neutral_prompt))
        else:
            with steer_residual(model, TARGET_LAYER, layer_vec, strength):
                print(generate_reply(model, tokenizer, neutral_prompt))

if __name__ == "__main__":
    main()
