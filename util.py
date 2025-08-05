import os
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Text 

import yaml
from transformers import PreTrainedTokenizer
from datasets import Dataset


INT_INFINITY = 2**63 - 1


@dataclass
class GenConfig:
    do_sample:      Optional[bool]  = True
    max_new_tokens: Optional[int]   = 512
    temperature:    Optional[float] = 0.7
    top_p:          Optional[float] = 1.0
    top_k:          Optional[int] = 50


def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def split_dataset_per_rank(dataset: Dataset, rank: int, world_size: int) -> Text:
    assert 1 <= rank <= world_size
    split_size = math.ceil(len(dataset) / world_size)
    dataset = dataset.select(range(
        (rank-1)*split_size, 
        min((rank)*split_size, len(dataset))
    ))
    return dataset


def get_output_path(output_dir: str, rank: int, world_size: int, suffix: str = "jsonl") -> Text:
    assert 1 <= rank <= world_size
    return os.path.join(output_dir, f"{str(rank).zfill(5)}-of-{str(world_size).zfill(5)}.{suffix}")