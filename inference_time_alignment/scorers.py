from dataclasses import dataclass, asdict
from typing import Text, List, Dict, Optional
from abc import ABC, abstractclassmethod

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from .utils import (
    SFTDataMapFunc, 
    SFTDataCollatorWithPadding,
    get_batch_logps,
    prepare_input
)


@dataclass
class ScorerInput:
    response: List[str]
    eos: List[bool]


@dataclass
class BaseScorer(ABC):
    
    @abstractclassmethod
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class ImplicitValueScorer(BaseScorer):
    model: PreTrainedModel
    ref_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: Optional[bool] = False
    model_prompt_template: Optional[str] = "{raw_prompt}"
    ref_model_prompt_template: Optional[str] = "{raw_prompt}"
    raw_prompt: Optional[str] = None

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self

    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        policy_all_logps = self.forward(
            self.model, 
            self.model_prompt_template, 
            input
        )
        ref_all_logps = self.forward(
            self.ref_model, 
            self.ref_model_prompt_template, 
            input
        )
        return policy_all_logps - ref_all_logps

    @torch.no_grad()
    def forward(
        self, 
        model: PreTrainedModel, 
        prompt_template: Text, 
        input: ScorerInput | Dict
    ) -> torch.Tensor:
        input = asdict(input)
        prompt = prompt_template.format(raw_prompt=self.raw_prompt)
        input["prompt"] = [prompt] * len(input["response"])

        tokens = SFTDataMapFunc(tokenizer=self.tokenizer, 
                                add_special_tokens=self.add_special_tokens)(input)
        batch  = SFTDataCollatorWithPadding(tokenizer=self.tokenizer)(
            [{k:v[i] for k,v in tokens.items()} for i in range(len(input["response"]))])
        batch = prepare_input(batch)

        all_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.to(torch.float32)

        return get_batch_logps(all_logits, batch["labels"])