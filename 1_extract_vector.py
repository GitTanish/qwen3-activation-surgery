import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
CONCEPT = "mood"
OUT_DIR = Path("artifacts")

def format_prompt(tokenizer, text:str) -> str:
    "wraps raw text in Qwen3's chat template with thinking disabled"
    messages = [{"role": "user", "content":text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

@torch.no_grad()
def compute_mean_activations(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    """Forward-pass prompts and return mean residual vector per layer: shape(num_layers, hidden_dim)"""
    layer_sums = None
    total_tokens= 0
    for text in prompts:
        formatted = format_prompt(tokenizer, text)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        # single forward pass
        outputs = model(**inputs, output_hidden_states=True)

        # hidden_state[0] is the initial token embedding, [1:] are the 28 transformer blocks
        stacked = torch.stack(outputs.hidden_states[1:], dim = 0) # (28, 1 , seq_len, D)
        stacked = stacked.squeeze(1).to(torch.float32)

        if layer_sums is None:
            layer_sums = stacked.sum(dim=1)
        else:
            layer_sums += stacked.sum(dim=1)

        total_tokens += stacked.shape[1]
    return layer_sums / total_tokens

def main():
    OUT_DIR.mkdir(exist_ok = True)
    data = json.loads(Path('prompts.json').read_text())[CONCEPT]

    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map = "auto"
    )
    model.eval()

    pos_mean = compute_mean_activations(model, tokenizer, data['positive'])
    neg_mean = compute_mean_activations(model, tokenizer, data['negative'])

    # difference of means: vector pointing from negative to positive each layer
    steering_matrix= pos_mean - neg_mean #(28, 1024) for smaller 0.6b model and (28,2048) for 1.8b model

    # check which layers show the strongest geometric seperation between the two concepts
    norms = steering_matrix.norm(dim=1)
    top_layers = torch.topk(norms, k=5).indices.tolist()

    out_file = OUT_DIR / f'{CONCEPT}_steering_matrix.pt'
    torch.save(steering_matrix, out_file)

    print(f"Model: {MODEL_ID} | Saved: {out_file} | shape: {tuple(steering_matrix.shape)}")


if __name__ == "__main__":
    main()
