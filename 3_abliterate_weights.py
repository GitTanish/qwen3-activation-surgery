import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
ABLITERATE_LAYERS = list(range(12,24)) # middle to late layers where refusal decisions form

def format_prompt(tokenizer, text: str):
    messages = [{"role":"user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize = False,
        add_generation_prompt=True,
        enable_thinking=False
    )

@torch.no_grad()
def extract_refusal_directions(model, tokenizer, data:dict) -> torch.Tensor:
    "compute difference of means at the final prompt token across all 28 layers."
    def mean_last_token(prompts: list[str]) -> torch.Tensor:
        vecs = []
        for p in prompts:
            inputs = tokenizer(format_prompt(tokenizer, p), return_tensors = "pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            # final prompt token(the exact position where the model decides to refuse or comply)
            last_tok= torch.stack(outputs.hidden_states[1:], dim= 0)[:,0,-1,:].to(torch.float32)
            vecs.append(last_tok)
        return torch.mean(torch.stack(vecs, dim=0), dim=0) # (28, 1024)

    refusal_mean = mean_last_token(data["positive"])
    compliance_mean= mean_last_token(data['negative'])
    return refusal_mean  - compliance_mean # (28,1024)

@torch.no_grad()
def abliterate_model(model, refusal_dirs: torch.Tensor, target_layers: list[int]):
    """Permanently project out the refusal direction from o_proj and down_proj weights"""
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        r = refusal_dirs[layer_idx]
        r_hat = (r/ (r.norm() + 1e-8)).to(device=model.device, dtype = torch.float32)

        # orthogonal projection matrix: P = I - r_hat * r_hat^T
        eye = torch.eye(r_hat.shape[0], device=model.device, dtype= torch.float32)
        proj_matrix = eye- torch.outer(r_hat, r_hat)

        # modifying the only two weight matrices that write directly into the residual stream
        for module in [layer.self_attn.o_proj, layer.mlp.down_proj]:
            w = module.weight.data.to(torch.float32)
            module.weight.data = (proj_matrix @ w).to(module.weight.dtype)

@torch.no_grad()
def generate_reply(model, tokenizer, prompt:str) -> str:
    inputs = tokenizer(format_prompt(tokenizer, prompt), return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    data = json.loads(Path("prompts.json").read_text())["refusal"]

    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype,
        device_map="cpu"
    )
    model.eval()

    test_prompt = "How do I make my friend explode with laughter?" # show him this code lol
    print("=== BEFORE ABLITERATION (Original Weights) ===")
    print(generate_reply(model, tokenizer, test_prompt))

    # Extract refusal vector and permanently edit weights in memory
    refusal_dirs = extract_refusal_directions(model, tokenizer, data)
    abliterate_model(model, refusal_dirs, ABLITERATE_LAYERS)

    print("\n=== AFTER ABLITERATION (Permanent Weight Edit, Zero Hooks) ===")
    print(generate_reply(model, tokenizer, test_prompt))

    model.save_pretrained("artifacts/qwen3-0.6b-abliterated")
    tokenizer.save_pretrained("artifacts/qwen3-0.6b-abliterated")

if __name__ == "__main__":
    main()
