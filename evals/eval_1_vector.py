import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
VECTOR_PATH = Path("artifacts/mood_steering_matrix.pt")

# Held-out prompts never seen during extraction
HELD_OUT = {
    "positive": [
        "Describe sitting alone in a cold hospital waiting room at midnight.",
        "Write about finding an old photo of someone who passed away years ago.",
        "Explain the quiet ache of watching everyone move on while you stay behind.",
        "Describe an empty playground rusting under gray winter rain.",
    ],
    "negative": [
        "Describe the energy of a packed stadium cheering after a last-second goal.",
        "Write about opening the windows on a crisp, sunny morning full of birdsong.",
        "Explain the excitement of starting a brand-new adventure with your best friends.",
        "Describe a lively family dinner full of laughter, warm food, and stories.",
    ],
}


def format_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


@torch.no_grad()
def project_prompts(model, tokenizer, prompts: list[str], unit_dirs: torch.Tensor) -> torch.Tensor:
    """Project mean-pooled hidden states onto unit steering directions: returns (num_prompts, 28)."""
    scores = []
    for p in prompts:
        inputs = tokenizer(format_prompt(tokenizer, p), return_tensors="pt").to(model.device)
        out = model(**inputs, output_hidden_states=True)
        hs = torch.stack(out.hidden_states[1:], dim=0)[:, 0].to(torch.float32)  # (28, seq, 1024)
        # Mean-pool over tokens to match 1_extract_vector.py
        pooled = hs.mean(dim=1)  # (28, 1024)
        # Dot product per layer: (28,)
        proj = (pooled * unit_dirs).sum(dim=-1)
        scores.append(proj)
    return torch.stack(scores, dim=0)  # (N, 28)


def main():
    steering = torch.load(VECTOR_PATH, map_location="cpu").to(torch.float32)  # (28, 1024)
    unit_dirs = steering / (steering.norm(dim=-1, keepdim=True) + 1e-8)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()

    pos_proj = project_prompts(model, tokenizer, HELD_OUT["positive"], unit_dirs)
    neg_proj = project_prompts(model, tokenizer, HELD_OUT["negative"], unit_dirs)

    # Compute Cohen's d across all 28 layers
    mean_diff = pos_proj.mean(dim=0) - neg_proj.mean(dim=0)
    pooled_std = torch.sqrt((pos_proj.var(dim=0) + neg_proj.var(dim=0)) / 2.0 + 1e-8)
    cohens_d = mean_diff / pooled_std

    top_vals, top_layers = torch.topk(cohens_d, k=5)
    print("=== Held-Out Generalization (Cohen's d) ===")
    for layer, d_val in zip(top_layers.tolist(), top_vals.tolist()):
        print(f"Layer {layer:2d} | Cohen's d = {d_val:+.2f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/eval_1_vector.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "vector_path": str(VECTOR_PATH),
                "cohens_d_per_layer": [round(x, 4) for x in cohens_d.tolist()],
                "top_5_layers": top_layers.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
