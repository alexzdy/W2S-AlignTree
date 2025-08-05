import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Text
import tyro
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available
from datasets import Dataset
import numpy as np

from utils import get_scorer, get_dataset 

import sys
sys.path.append("../")

from util import (
    set_seeds, split_dataset_per_rank,
    get_output_path, GenConfig
)

from inference_time_alignment.decoders.mcts import MCTSGenerationMixin
from inference_time_alignment.utils import extract_responses


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#
@dataclass
class MCTSGenConfig:
    uct_constant: Optional[float] = 1.5  # UCT exploration constant, c
    num_candidates: Optional[int] = 3  # Number of candidates, K
    num_iterations: Optional[int] = 200  # Maximum number of iterations, m
    l: Optional[int] = 1  # Chunk length, L
    entropy_weight: Optional[float] = 0.2  # Entropy weight for exploration, w

    others: GenConfig = field(default_factory=lambda: GenConfig(max_new_tokens=50, temperature=0.7, top_p=1.0, top_k=50))  # generation config

    def __post_init__(self):
        pass


@dataclass
class ScriptArguments:
    model_name:            Optional[Text] = "/path/to/strong_models"

    base_prompt_template:  Optional[str]  = "Here is a movie review from imdb: {raw_prompt}"
    dataset_name:          Optional[Text] = "ZHZisZZ/imdb_preference"

    output_dir:            Optional[Text] = "/path/to/results/gen"
    overwrite:             Optional[bool] = True
    rank:                  Optional[int]  = 1  # one-based indexing
    world_size:            Optional[int]  = 1
    seed:                  Optional[int]  = 1
    load_in_4bit:          Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True
    gen:  MCTSGenConfig = field(default_factory=lambda: MCTSGenConfig())

script_args = tyro.cli(ScriptArguments)
print(script_args)
set_seeds(script_args.seed)

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#
# Load dataset
print(f"loading dataset {script_args.dataset_name} ...")
dataset = get_dataset(script_args.dataset_name)
# split dataset by rank and append rank suffix to output path, e.g., "00001-00008.jsonl"
dataset = split_dataset_per_rank(dataset, script_args.rank, script_args.world_size)
output_path = get_output_path(script_args.output_dir, script_args.rank, script_args.world_size)
# skip if previous generation result exists and we do not want to overwrite it
if os.path.exists(output_path) and not script_args.overwrite: exit()


# Load base model, tokenizer, and scorer
print(f"loading base model {script_args.model_name} ...")
base = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=script_args.load_in_4bit) if is_bitsandbytes_available() else None,
    attn_implementation="flash_attention_2" if script_args.use_flash_attention_2 and is_flash_attn_2_available() else None,
)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

# get mcts model
print("setting up MCTS model...")
mcts_model = MCTSGenerationMixin(base, tokenizer)
# get scorer
print("loading scorer...")
scorer = get_scorer(
    load_in_4bit=script_args.load_in_4bit,
    use_flash_attention_2=script_args.use_flash_attention_2,
)

#-----------------------------------------------------------------------------#
#---------------------------------- search -----------------------------------#
#-----------------------------------------------------------------------------#
results = []
for idx, raw_prompt in enumerate(tqdm.tqdm(dataset["raw_prompt"])):
    prompt = script_args.base_prompt_template.format(raw_prompt=raw_prompt)
    prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
       
    outputs = mcts_model.search(
        input_ids=prompt_tokenized["input_ids"].cuda(),
        attention_mask=prompt_tokenized["attention_mask"].cuda(),
        scorer=scorer.set_raw_prompt(raw_prompt), 
        num_candidates=script_args.gen.num_candidates,
        num_iterations=script_args.gen.num_iterations, 
        uct_constant=script_args.gen.uct_constant, 
        l=script_args.gen.l,
        entropy_weight=script_args.gen.entropy_weight,
        **asdict(script_args.gen.others), 
    )

    response = extract_responses(outputs, tokenizer, prompt=prompt)[0]
    results.append({
        "prompt": raw_prompt,
        "response": response,
    })
    
    print(f"Generating response: {response}")

dataset = Dataset.from_list(results)
dataset.to_json(output_path)
