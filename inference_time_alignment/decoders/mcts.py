from dataclasses import dataclass, field
from typing import Dict, Any, Text, Optional, Union, List
import math
import heapq  # min-heap, progressive increase from min to max

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationMixin
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    EosTokenCriteria,
    StoppingCriteriaList
)

from inference_time_alignment.scorers import BaseScorer, ScorerInput
from inference_time_alignment.utils import (
    StopOnStringCriteria,
    extract_responses,
    get_truncated_responses
)


@dataclass
class MCTSNode:
    """
    MCTS Node
    """
    state: torch.Tensor
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    score: float = 0.0
    gen_score: float = -float('inf')  # Score of the generation node
    last_token: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None  # Store past_key_values for each node
    prior_prob: float = 0.0  # New: Store prior probability for PUCT
    entropy: float = 0.0  # New: Store entropy for exploration

    def puct_value(self, parent_visits: int, uct_constant: float = 1.0, gen_flag: bool = False) -> float:
        if self.visits == 0:
            return float('inf')
        if gen_flag:
            return self.gen_score + uct_constant * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visits)
        return self.score + uct_constant * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visits)

    def puct_with_entropy(self, parent_visits: int, uct_constant: float = 1.0, entropy_weight: float = 0.2, gen_flag: bool = False) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploitation_term = self.gen_score if gen_flag else self.score 
        
        exploration_term = uct_constant * self.prior_prob * (math.sqrt(parent_visits) / (1 + self.visits))
        
        if self.parent:
            entropy_bonus = self.parent.entropy * entropy_weight
        else:
            entropy_bonus = 0.0
        
        return exploitation_term + exploration_term * (1 + entropy_bonus)

    def uct_value(self, parent_visits: int, uct_constant: float = 2.0) -> float:
        if self.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return self.score + uct_constant * math.sqrt(math.log(parent_visits) / (1 + self.visits))


@dataclass
class MCTSGenerationMixin(GenerationMixin):
    base: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __getattribute__(self, name: Text) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.base, name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        return self.base.prepare_inputs_for_generation(input_ids, **model_kwargs)

    @torch.no_grad()
    def search(
        self,
        input_ids: torch.LongTensor,
        scorer: BaseScorer,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        eos_strings: Optional[List[str]] = None,
        split_by_prompt_text: Optional[bool] = True,
        num_candidates: Optional[int] = 5,
        num_iterations: Optional[int] = 100,
        uct_constant: Optional[float] = 2.0,
        entropy_weight: Optional[float] = 0.2,
        epsilon: Optional[float] = 1e-10, 
        num_select: Optional[int] = 10,
        l: Optional[int] = 5,
        **kwargs,
    ):
        if not self.generation_config.pad_token_id:
            self.generation_config.pad_token_id = self.generation_config.eos_token_id
            if isinstance(self.generation_config.pad_token_id, list):
                self.generation_config.pad_token_id = self.generation_config.pad_token_id[0]

        logits_warper = LogitsProcessorList()
        if temperature:
            logits_warper.append(TemperatureLogitsWarper(temperature))
        if top_k:
            logits_warper.append(TopKLogitsWarper(top_k))
        if top_p:
            logits_warper.append(TopPLogitsWarper(top_p))

        stopping_criteria = StoppingCriteriaList()
        if eos_strings:
            stopping_criteria.extend([
                StopOnStringCriteria(input_ids.size(1), eos_string, self.tokenizer)
                for eos_string in eos_strings
            ])
        if max_length:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_new_tokens:
            stopping_criteria.append(MaxLengthCriteria(max_length=input_ids.size(1) + max_new_tokens))
        if self.generation_config.eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=self.generation_config.eos_token_id))

        return self._search(
            input_ids=input_ids,
            scorer=scorer,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            eos_strings=eos_strings,
            split_by_prompt_text=split_by_prompt_text,
            num_candidates=num_candidates,
            uct_constant=uct_constant,
            entropy_weight=entropy_weight, 
            num_iterations=num_iterations,
            epsilon=epsilon,
            num_select=num_select,
            l=l,
            **kwargs
        )

    @torch.no_grad()
    def _search(
        self,
        input_ids: torch.LongTensor,
        scorer: BaseScorer,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        eos_strings: Optional[List[str]] = None,
        split_by_prompt_text: Optional[bool] = True,
        num_candidates: Optional[int] = 5,
        uct_constant: Optional[float] = 2.0,
        entropy_weight: Optional[float] = 0.2,
        pad_token_id: Optional[int] = None,
        return_dict_in_generate: Optional[bool] = None,
        num_iterations: Optional[int] = 100,
        epsilon: Optional[float] = 1e-10,
        num_select: Optional[int] = 10,
        l: Optional[int] = 5,
        **model_kwargs,
    ) -> Union[Dict[str, Any], torch.LongTensor]:
        
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        device = input_ids.device
        root_node = MCTSNode(
            state=input_ids.clone(),
            prior_prob=1.0
        )

        prompt_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        prompt_len = input_ids.size(1)

        def _calculate_entropy(probs: torch.Tensor, epsilon: float = 1e-9) -> float:
            """Calculates Shannon entropy H(P) = -sum(p * log2(p)) to measure randomness."""
            log_probs = torch.log2(probs + epsilon)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            return entropy.item()

        def expand_node(node: MCTSNode, num_expand: int, l: int):
            input_ids_template = node.state.repeat(num_expand, 1)
            log_priors_accumulated = torch.zeros(num_expand, device=device)
            unfinished_sequences = torch.ones(num_expand, dtype=torch.long, device=device)
            
            current_input_ids = input_ids_template
            
            for _l in range(l):
                outputs = self.base(input_ids=current_input_ids, use_cache=False, return_dict=True)
                next_token_logits = outputs.logits[:, -1, :]
                
                next_token_logits = logits_warper(current_input_ids, next_token_logits)
                probs = nn.functional.softmax(next_token_logits, dim=-1)


                if _l == 0:
                    node.entropy = _calculate_entropy(probs[0])
                    first_sequence_probs = probs[0].unsqueeze(0)
                    next_tokens = torch.multinomial(first_sequence_probs, num_samples=num_expand).squeeze(0)
                    picked_priors = first_sequence_probs.repeat(num_expand, 1).gather(1, next_tokens.unsqueeze(1)).squeeze(1)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    picked_priors = probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

                log_priors_accumulated += torch.log(picked_priors + epsilon) * unfinished_sequences

                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                
                current_input_ids = torch.cat([current_input_ids, next_tokens[:, None]], dim=-1)
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(current_input_ids, None)

                if unfinished_sequences.max() == 0:
                    break

            effective_l = _l + 1
            avg_priors = torch.exp(log_priors_accumulated / effective_l)
            
            res = []
            for _i in range(num_expand):
                child_node = MCTSNode(
                    state=current_input_ids[_i, :].unsqueeze(0),
                    last_token=current_input_ids[_i, -1],
                    parent=node,
                    visits=0,
                    prior_prob=avg_priors[_i].item()
                )
                res.append(child_node)
            return res

        for iteration in range(num_iterations):
            node = root_node
            path = [root_node]

            # === Selection ===
            while True:
                if stopping_criteria(node.state, None) or not node.children:
                    break
                
                uct_values = [
                    child.puct_with_entropy(
                        parent_visits=node.visits, 
                        uct_constant=uct_constant, 
                        entropy_weight=entropy_weight
                    )
                    for child in node.children
                ]
                
                best_index = int(torch.tensor(uct_values).argmax().item())
                node = node.children[best_index]
                path.append(node)
            
            # === Expansion ===
            if not stopping_criteria(node.state, None):
                children = expand_node(node, num_candidates, l)
                
                # === Evaluation ===
                active_children = [
                    child for child in children
                    if not stopping_criteria(child.state, None)
                ]

                if active_children:
                    batch_states = torch.cat([child.state for child in active_children], dim=0)
                    if split_by_prompt_text:
                        batch_responses = extract_responses(batch_states, tokenizer=self.tokenizer, prompt=prompt_str)
                    else:
                        batch_responses = extract_responses(batch_states, tokenizer=self.tokenizer, prompt_len=prompt_len)

                    if eos_strings:
                        batch_responses, _ = get_truncated_responses(batch_responses, eos_strings)

                    batch_eos = torch.zeros(len(batch_responses), dtype=torch.bool, device=device)
                    batch_scores = scorer(ScorerInput(response=batch_responses, eos=batch_eos)).view(-1)

                    for child, score in zip(active_children, batch_scores):
                        child.score = score.item()
                        child.gen_score = score.item()
                        child.visits += 1
                
                for child in children:
                    if stopping_criteria(child.state, None):
                        child.score = -float("inf")
                        child.gen_score = -float("inf")
                        child.visits += 1

                node.children.extend(children)

            # === Backpropagation ===
            for node_in_path in reversed(path):
                node_in_path.visits += 1
                if node_in_path.children:
                    node_in_path.score = max([child.score for child in node_in_path.children])


        # After all iterations, select best candidate
        def select_final_node(root: MCTSNode) -> MCTSNode:
            select_candidates = []  # (uct, id, node)
            max_score_node = root

            max_score_node.score = -float("inf")

            def traverse(node: MCTSNode):
                nonlocal max_score_node

                if stopping_criteria(node.state, None):
                    return

                if node.children:
                    # print(node.score,max_score_node.score)
                    for child in node.children:
                        traverse(child)
                    
                    if node.score > max_score_node.score:
                        max_score_node = node 
                    
                    if node.parent and any(stopping_criteria(c.state, None) for c in node.children):
                        # Calculate UCT value and push to the heap
                        uct = node.puct_value(node.parent.visits, uct_constant, True)
                        heapq.heappush(select_candidates, (uct, id(node), node))  # min-heap, progressive increase from min to max

                        # Maintain the heap size to num_select, remove min (the first)
                        if len(select_candidates) > num_select:
                            heapq.heappop(select_candidates)
                
            traverse(root)

            # Update scores of candidate nodes and their children
            if not select_candidates:
                print("no node stopping_criteria")
            
                select_candidates.append((0, id(max_score_node), max_score_node))
                stop_flag = torch.tensor([False], device=device)
                
            else:
                stop_flag = torch.tensor([True], device=device)

            # Update scores of candidate nodes and their children
            final_node = root  # Initialize with the root node
            final_node.score = -float("inf")
            for _, _, node in select_candidates:
                for child in node.children:
                    if split_by_prompt_text:
                        responses = extract_responses(child.state, tokenizer=self.tokenizer, prompt=prompt_str)
                    else:
                        responses = extract_responses(child.state, tokenizer=self.tokenizer, prompt_len=prompt_len)
                    
                    if eos_strings:
                        responses, _ = get_truncated_responses(responses, eos_strings)

                    scores = scorer(ScorerInput(response=responses, eos=stop_flag))
                    score = scores.item()
                    child.score = score

                    if child.score > final_node.score:
                        final_node = child
            
            return final_node

        final_node = select_final_node(root_node)
        output_ids = final_node.state
        
        print(f"Final length: {output_ids.shape[1] - prompt_len}")
        
        if return_dict_in_generate:
            return {
                "output_ids": output_ids,
                "scores": [child.score for child in final_node.parent.children] if final_node.parent else []
            }
        else:
            return output_ids