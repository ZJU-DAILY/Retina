from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import torch
import gc
from PIL import Image
from tqdm import tqdm
from transformers import BatchEncoding, BatchFeature
import numpy as np
from collections import defaultdict
import numba
from numba import types
from numba.typed import Dict, List as NumbaList
from colpali_engine.utils.torch_utils import get_torch_device

@numba.njit(nogil=True, parallel=True, cache=True)
def numba_score_float(inverted_index_ids, inverted_index_floats, query_terms, query_values, threshold, size_collection):
    scores = np.zeros(size_collection, dtype=np.float32)
    n = len(query_terms)
    for _idx in range(n):
        term = query_terms[_idx]
        query_val = query_values[_idx]
        if term in inverted_index_ids:
            retrieved_indexes = inverted_index_ids[term]
            retrieved_floats = inverted_index_floats[term]
            for j in range(len(retrieved_indexes)):
                doc_id = retrieved_indexes[j]
                doc_val = retrieved_floats[j]
                scores[doc_id] += query_val * doc_val
    # Apply threshold
    filtered_indexes = np.nonzero(scores > threshold)[0]
    return filtered_indexes, scores[filtered_indexes]

class BaseVisualRetrieverProcessor(ABC):
    """
    Base class for visual retriever processors.
    """

    @abstractmethod
    def process(
        self,
        input: list,
        examples: list,
        is_query: bool = False,
        use_example: bool = False,
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    # @abstractmethod
    # def process_queries(
    #     self,
    #     queries: List[str],
    #     max_length: int = 50,
    #     suffix: Optional[str] = None,
    # ) -> Union[BatchFeature, BatchEncoding]:
    #     pass

    @abstractmethod
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    # @staticmethod
    # def score_single_vector(
    #     qs: List[torch.Tensor],
    #     ps: List[torch.Tensor],
    #     batch_size: int = 128,
    #     device: Optional[Union[str, torch.device]] = None,
    # ) -> torch.Tensor:
    #     """
    #     Compute the dot product score for the given single-vector query and passage embeddings.
    #     """
    #     device = device or get_torch_device("auto")

    #     if len(qs) == 0:
    #         raise ValueError("No queries provided")
    #     if len(ps) == 0:
    #         raise ValueError("No passages provided")

    #     qs_stacked = torch.stack(qs).to(device)
    #     ps_stacked = torch.stack(ps).to(device)

    #     scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
    #     assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

    #     scores = scores.to(torch.float32)
    #     return scores
    @staticmethod
    def score_single_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []
        ps_stacked = torch.stack(ps).to(device)  # 形状: (num_passages, dim)
        for start_idx in tqdm(range(0, len(qs), batch_size), desc="Processing Queries and Compute Score.....", unit="batch"):
            end_idx = min(start_idx + batch_size, len(qs))
            batch_qs = qs[start_idx:end_idx]
            qs_stacked = torch.stack(batch_qs).to(device)  # 形状: (batch_size, dim)
            batch_scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
            batch_scores = batch_scores.to(torch.float32)
            scores_list.append(batch_scores.cpu())
        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        return scores
    @staticmethod
    def score_multi_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in tqdm(range(0, len(qs), batch_size), desc="Processing Queries and Compute Score.....", unit="batch"):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_sparse_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
        threshold: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        device = device or get_torch_device("auto")
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        ps_len = len(ps)
        # 初始化 Numba typed.Dict
        inverted_index_ids = Dict.empty(
            key_type=types.int32,
            value_type=types.ListType(types.int32),
        )
        inverted_index_floats = Dict.empty(
            key_type=types.int32,
            value_type=types.ListType(types.float32),
        )
        for doc_id, p in tqdm(enumerate(ps), desc="Building Inverted Index", total=ps_len):
            p = p.to(device)
            # indices = p.nonzero(as_tuple=True)[0].tolist()
            # values = p[indices].tolist()
            indices = p._indices().tolist()[0]  # 获取非零元素的索引
            values = p._values().tolist()  # 获取非零元素的值
            for term, value in zip(indices, values):
                if value > 0:
                    term = int(term)
                    doc_id_int = int(doc_id)
                    value_float = float(value)
                    if term not in inverted_index_ids:
                        inverted_index_ids[term] = NumbaList.empty_list(types.int32)
                        inverted_index_floats[term] = NumbaList.empty_list(types.float32)
                    inverted_index_ids[term].append(doc_id_int)
                    inverted_index_floats[term].append(value_float)
        del ps
        gc.collect()
        scores_list = []

        for start_idx in tqdm(range(0, len(qs), batch_size), desc="Processing Queries and Compute Score.....", unit="batch"):
            end_idx = min(start_idx + batch_size, len(qs))
            batch_qs = qs[start_idx:end_idx]

            batch_qs = [q.to(device) for q in batch_qs]
            batch_qs_terms = []
            batch_qs_values = []
            for q in batch_qs:
                # indices = q.nonzero(as_tuple=True)[0].tolist()
                # values = q[indices].tolist()
                indices = q._indices().tolist()[0]  # 获取非零元素的索引
                values = q._values().tolist()  # 获取非零元素的值
                batch_qs_terms.append(indices)
                batch_qs_values.append(values)

            batch_scores = []
            for q_terms, q_values in zip(batch_qs_terms, batch_qs_values):
                q_terms_np = np.array(q_terms, dtype=np.int32)
                q_values_np = np.array(q_values, dtype=np.float32)
                filtered_indexes, scores = numba_score_float(
                    inverted_index_ids,
                    inverted_index_floats,
                    q_terms_np,
                    q_values_np,
                    threshold=threshold, 
                    size_collection=ps_len
                )
                score_tensor = torch.zeros(ps_len, dtype=torch.float32)
                score_tensor[filtered_indexes] = torch.from_numpy(scores)
                batch_scores.append(score_tensor)

            scores_batch_tensor = torch.stack(batch_scores, dim=0)
            scores_list.append(scores_batch_tensor.cpu())

        scores = torch.cat(scores_list, dim=0)
        
        return scores
    @abstractmethod
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 14,
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
        image of size (height, width) with the given patch size.
        """
        pass
