import os
from dataclasses import dataclass
from typing import Optional, Text

import tyro
import tqdm
import torch
from datasets import Dataset, load_dataset
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.append("../")

from util import set_seeds

set_seeds(1)


@dataclass
class ScriptArguments:

    generation_dir:  Optional[Text] = "/path/to/results/gen"
    evaluation_dir:  Optional[Text] = "/path/to/results/eval"


script_args = tyro.cli(ScriptArguments)
generation = load_dataset(script_args.generation_dir, split="train")

# local model
model_path = "lvwerra/distilbert-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda:0")
rm = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0, function_to_apply="none", return_all_scores=True)

results = []
with torch.no_grad():
    for sample in tqdm.tqdm(generation):
        rm_output = rm(sample["prompt"] + sample["response"])[0]

        assert rm_output[1]["label"] == "POSITIVE"
        # log_p positive - log_p negative
        score = rm_output[1]["score"] - rm_output[0]["score"]
        results.append({
            "prompt": sample["prompt"],
            "response": sample["response"],
            "score": score,
        })

# raw
dataset = Dataset.from_list(results)
dataset.to_json(os.path.join(script_args.evaluation_dir, "raw.jsonl"))

# mean
scores = [result["score"] for result in results]
mean_score = sum(scores) / len(scores)
print(f"Mean score: {mean_score:.4f}")
with open(os.path.join(script_args.evaluation_dir, "mean.txt"), "w") as f:
    f.write(f"{mean_score:.4f}")
