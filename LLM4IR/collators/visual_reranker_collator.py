from typing import Any, Dict, List, Union, cast
import torch
from PIL.Image import Image
import random
from LLM4IR.models.idefics_2 import ColIdefics2Processor
from LLM4IR.models.paligemma import ColPaliProcessor
from LLM4IR.utils.processing_utils import BaseVisualRetrieverProcessor
from LLM4IR.collators.visual_retriever_collator import VisualRetrieverCollator

class VisualRerankerCollator(VisualRetrieverCollator):
    """
    Collator for training vision retrieval models.
    """
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        neg_num: int = 3
    ):
        super().__init__(processor, max_length, neg_num)
        
        self.processor = processor
        self.image_token_id = None
        self.image_token = '<image>'
        self.query_prefix = "Query: "
        self.candidate_prefix = "Candidate: "
        self.max_length = max_length
        self.neg_num = neg_num

        if isinstance(self.processor, ColPaliProcessor) or isinstance(self.processor, ColIdefics2Processor):
            self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                self.processor.tokenizer.additional_special_tokens.index("<image>")
            ]

        if isinstance(self.processor, ColPaliProcessor):
            if self.processor.tokenizer.padding_side != "right":
                print("Setting padding side to right")
                self.processor.tokenizer.padding_side = "right"


    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Collate function for the vision retriever associated to the collator's processor.
        """
        querys: Union[List[dict[str, Union[str, Image]]], List[None]] = []  # some documents don't have a query
        pos_cand: Union[List[dict[str, Union[str, Image]]], List[None]]= []
        pos_pairs:Union[List[dict[str, Union[str, Image]]], List[None]] = []
        neg_pairs: Union[List[dict[str, Union[str, Image]]], List[None]] = []
        def add_image_token(example_key: dict) -> dict:
            if isinstance(example_key.get('img'), list):
                image_token_count = len([img for img in example_key['img'] if img is not None])
            else:
                image_token_count = 1 if example_key['img'] is not None else 0
                example_key['img'] = [example_key['img']] if example_key['img'] is not None else []
            txt_with_image_tokens = f"{(self.image_token) * image_token_count}{example_key['txt']}"
            example_key['txt'] = txt_with_image_tokens
            return example_key
        def get_pairs(query: dict, candidate: dict, prompt: str = '') -> list:
            example_txt = self.query_prefix + query['txt'] + "\n" + self.candidate_prefix + candidate['txt']
            example_img = [img for img in query['img'] if img is not None] + [img for img in candidate['img'] if img is not None]
            fianl_example = {'txt': example_txt, 'img': example_img, 'prompt': prompt}
            return fianl_example

        if self.processor is None or not isinstance(self.processor, BaseVisualRetrieverProcessor):
            raise ValueError("Processor should be provided for vision collator.")

        for example in examples:
            
            #TODO: add prompt
            default_prompt = ["Given a Query and a Candidate, determine whether the candidate contains an answer to the query by providing a prediction of either 'Yes' or 'No'.\n", '']
            prompt = example["prompt"] if ('prompt' in example and example['prompt']) else default_prompt[0]
            
            if "pos_cand" in example and example["pos_cand"] is not None:
                pos_cand.append(add_image_token(example["pos_cand"]))

            if "query" in example and example["query"] is not None:
                querys.append(add_image_token(example["query"]))
                
            pos_pairs.append(get_pairs(example["query"], example["pos_cand"], prompt))
                
            if "neg_cand_list" in example and example["neg_cand_list"] is not None and len(example["neg_cand_list"]) > 0:
                for neg in example["neg_cand_list"][:self.neg_num]:
                    neg_pairs.append(get_pairs(example["query"], neg, prompt))
        
        if len(neg_pairs) < len(examples):
            for i, query in enumerate(querys):
                sample_num = min(self.neg_num, len(pos_cand)-1)
                neg_cand_list = random.sample(pos_cand[:i] + pos_cand[i+1:], sample_num)
                for neg_cand in neg_cand_list:
                    neg_pairs.append(get_pairs(query, neg_cand, prompt))
        
        batch_pos_pairs, batch_neg_pairs, batch_combined_pairs = None, None, None
        
        if len(pos_pairs) > 0:
            batch_pos_pairs = self.processor.process(
                input = pos_pairs)
        
        if len(neg_pairs) > 0:
            batch_neg_pairs = self.processor.process(
                input = neg_pairs)
            
        combined_pairs = pos_pairs + neg_pairs
        batch_combined_pairs = self.processor.process(
            input = combined_pairs)
        
        pos_pairs_label = torch.full((len(pos_pairs),), self.processor.Yes_token_id, dtype=torch.long)
        neg_pairs_label = torch.full((len(neg_pairs),), self.processor.No_token_id, dtype=torch.long)
        combined_pairs_label = torch.cat((pos_pairs_label, neg_pairs_label), dim=0)
        
        del pos_pairs, neg_pairs, combined_pairs, querys, pos_cand
        # Prefix each key with "doc_" or "neg_doc_" to avoid key conflicts
        
        batch_all = {}
        batch_combined_pairs = {f"combined_{k}": v for k, v in batch_combined_pairs.items()}
        batch_all.update(batch_combined_pairs)
        batch_all["combined_label"] = combined_pairs_label
        del batch_combined_pairs, 
        
        if batch_pos_pairs is not None:
            batch_pos_pairs = {f"doc_{k}": v for k, v in batch_pos_pairs.items()}
            batch_all.update(batch_pos_pairs)
            batch_all["doc_label"] = pos_pairs_label
            del batch_pos_pairs, pos_pairs_label
        if batch_neg_pairs is not None:
            batch_neg_pairs = {f"neg_doc_{k}": v for k, v in batch_neg_pairs.items()}
            batch_all.update(batch_neg_pairs)
            batch_all["neg_doc_label"] = neg_pairs_label
            del batch_neg_pairs, neg_pairs_label
        return batch_all