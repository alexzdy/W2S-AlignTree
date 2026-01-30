import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dataclasses import dataclass
from typing import Optional, Text
import tyro
import tqdm
import torch
import numpy as np
from datasets import Dataset, load_dataset
from safetensors.torch import load_file
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class ScriptArguments:

    generation_dir: Optional[Text] = "/path/to/results/gen"
    evaluation_dir: Optional[Text] = "/path/to/results/eval"



class RewardModelWrapper:
    """Wrapper class for loading and using the Llama2-based reward model"""

    def __init__(self, model_path="meta-llama/Llama-2-7b-hf"):
        # Initialize device and model configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load base model with bfloat16 precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Configure tokenizer with proper padding
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LoRA adapter weights
        lora_path = "/path/to/summary_rm_lora"  # 
        self.model = PeftModel.from_pretrained(
            self.model, lora_path, adapter_name="lora_adapter", is_trainable=False
        )

        # Initialize projection head with float32 precision
        self.projection = torch.nn.Linear(
            self.model.config.hidden_size, 1, dtype=torch.float32
        ).to(self.device)

        # Load projection head weights from safetensors
        self._load_projection_weights(
            "/path/to/summary_rm_lora/value_head.safetensors"
        )

    def _load_projection_weights(self, safetensors_path):
        """Load weights for the projection head from safetensors file"""
        loaded_tensors = load_file(safetensors_path)
        weight_mapping = {
            "v_head.summary.weight": "weight",
            "v_head.summary.bias": "bias",
        }
        state_dict = {weight_mapping[k]: v for k, v in loaded_tensors.items()}
        self.projection.load_state_dict(state_dict)

    def get_rewards(self, texts):
        """Calculate reward scores for a batch of text inputs"""
        # Tokenize input texts with proper padding and truncation
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(self.device)

        # Forward pass with mixed precision
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                output_hidden_states=True,
            )

        # Extract final hidden states and calculate rewards
        last_hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32)
        return self.projection(last_hidden).squeeze(-1).cpu().detach().numpy()


def evaluate_generation(reward_model, generation_file, filename, evaluation_dir):
    print(f"Processing file: {generation_file}")
    # Load generated summaries
    generation_data = load_dataset('json', data_files=generation_file, split="train")

    # Process data in batches for efficiency
    batch_size = 8  # Adjust based on GPU memory
    results = []

    for i in tqdm.trange(0, len(generation_data), batch_size):
        batch = generation_data[i: i + batch_size]

        # Combine prompts and responses with proper formatting
        combined_texts = [
            f"{prompt}{response}"  # Maintain training format
            for prompt, response in zip(batch["prompt"], batch["response"])
        ]

        # Calculate reward scores for current batch
        batch_scores = reward_model.get_rewards(combined_texts)

        # Store results with original data
        for j, score in enumerate(batch_scores):
            results.append(
                {
                    "prompt": batch["prompt"][j],
                    "response": batch["response"][j],
                    "score": float(score),
                }
            )

    # Save raw evaluation results
    output_dataset = Dataset.from_list(results)
    output_path = os.path.join(evaluation_dir, f"{filename}_eval_raw.jsonl")
    output_dataset.to_json(output_path)

    # Calculate and save mean score
    mean_score = np.mean([r["score"] for r in results])
    mean_file_path = os.path.join(evaluation_dir, f"{filename}_eval_mean.txt")
    with open(mean_file_path, "w") as f:
        f.write(f"Mean Reward Score: {mean_score:.4f}\n")
        f.write(f"Evaluated Samples: {len(results)}\n")

    return mean_score


def main(script_args: ScriptArguments):
    # Initialize reward model
    reward_model = RewardModelWrapper()

    for filename in os.listdir(script_args.generation_dir):
        if filename.endswith('.jsonl'):
            try:
                generation_file = os.path.join(script_args.generation_dir, filename)
                filename = filename.split(".jsonl")[0]
                mean_score = evaluate_generation(reward_model, generation_file, filename, script_args.evaluation_dir)
                print(f"File: {filename}, Mean Reward Score: {mean_score:.4f}")
            except ValueError:
                print(f"Could not extract valid steps from filename {filename}. Skipping...")


if __name__ == "__main__":
    args = tyro.cli(ScriptArguments)
    main(args)