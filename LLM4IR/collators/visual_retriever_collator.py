from typing import Any, Dict, List, Union, cast

from PIL.Image import Image

from LLM4IR.models.idefics_2 import ColIdefics2Processor
from LLM4IR.models.paligemma import ColPaliProcessor
from LLM4IR.utils.processing_utils import BaseVisualRetrieverProcessor


class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        use_example: bool = False
    ):
        self.processor = processor
        self.image_token_id = None
        self.image_token = '<image>'
        self.query_prefix = "Query: "
        self.candidate_prefix = "Candidate: "
        self.max_length = max_length
        self.use_example = use_example

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
        # Placeholders
        querys: Union[List[dict[str, Union[str, Image]]], List[None]] = []  # some documents don't have a query
        pos_cand: Union[List[dict[str, Union[str, Image]]], List[None]]= []
        neg_cand: Union[List[dict[str, Union[str, Image]]], List[None]] = []
        def add_image_token(example_key: dict, prompt: str = '') -> dict:
            if isinstance(example_key.get('img'), list):
                image_token_count = len([img for img in example_key['img'] if img is not None])
            else:
                image_token_count = 1 if example_key['img'] is not None else 0
                example_key['img'] = [example_key['img']] if example_key['img'] is not None else []
            txt_with_image_tokens = f"{(self.image_token) * image_token_count}{example_key['txt']}"
            example_key['txt'] = txt_with_image_tokens
            example_key['prompt'] = prompt
            return example_key
        def get_in_batch_example(example: dict) -> list:
            query = example["query"]
            pos = example["pos_cand"]
            example_txt = self.query_prefix + query['txt'] + "\n" + self.candidate_prefix + pos['txt']
            example_img = [img for img in query['img'] if img is not None] + [img for img in pos['img'] if img is not None]
            fianl_example = {'txt': example_txt, 'img': example_img, 'dataset_id': example['dataset_id'], 'prompt': example['prompt']}
            return fianl_example

        if self.processor is None or not isinstance(self.processor, BaseVisualRetrieverProcessor):
            raise ValueError("Processor should be provided for vision collator.")

        # Process each example
        in_batch_example = []
        for example in examples:
            
            prompt = example["prompt"] if ('prompt' in example and example['prompt']) else ''
            
            if "pos_cand" in example and example["pos_cand"] is not None:
                pos_cand.append(add_image_token(example["pos_cand"], prompt))

            if "query" in example and example["query"] is not None:
                querys.append(add_image_token(example["query"], prompt))
                
            if "neg_cand_list" in example and example["neg_cand_list"] is not None and len(example["neg_cand_list"]) > 0:
                for neg in example["neg_cand_list"]:
                    neg_cand.append(add_image_token(neg, prompt))
               
            if self.use_example:
                in_batch_example.append(get_in_batch_example(example))

        batch_query, batch_pos_cand, batch_neg_cand = None, None, None
        
        if len(querys) > 0:
            batch_query = self.processor.process(
                input = querys, examples=in_batch_example, is_query=True, use_example=self.use_example)
        
        if len(neg_cand) > 0:
            batch_neg_cand = self.processor.process(
                input = neg_cand, is_query=False)
            
        if len(pos_cand) > 0:
            batch_pos_cand = self.processor.process(
                input = pos_cand, is_query=False)
        
        # Prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_all = {}
        # batch_all = {f"doc_{k}": v for k, v in batch_pos_cand.items()}
        if batch_pos_cand is not None:
            batch_pos_cand = {f"doc_{k}": v for k, v in batch_pos_cand.items()}
            batch_all.update(batch_pos_cand)
            del batch_pos_cand
        if batch_query is not None:
            batch_query = {f"query_{k}": v for k, v in batch_query.items()}
            batch_all.update(batch_query)
            del batch_query
        if batch_neg_cand is not None:
            batch_neg_cand = {f"neg_doc_{k}": v for k, v in batch_neg_cand.items()}
            batch_all.update(batch_neg_cand)
            del batch_neg_cand
        
        return batch_all