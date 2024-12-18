# Standard library
import base64
import io
import json
import os
import random
from enum import Enum
from typing import Callable, List, Union, Any

# Third-party
import torch
from PIL import Image

from collections import defaultdict
from torch.utils.data import Dataset

DATASET_CAN_NUM_UPPER_BOUND = 10000000  # Maximum number of candidates per dataset
DATASET_QUERY_NUM_UPPER_BOUND = 500000  # Maximum number of queries per dataset

MBEIR_TASK = {
    "text -> image": 0,
    "text -> text": 1,
    "text -> image,text": 2,
    "image -> text": 3,
    "image -> image": 4,
    "image -> text,image": 5,  # This is not a valid task For now, we will ignore this task
    "image,text -> text": 6,
    "image,text -> image": 7,
    "image,text -> image,text": 8,
}
def get_mbeir_task_id(source_modality, target_modality):
    """Get the MBEIR task ID using source and target modalities."""
    task_name = f"{source_modality} -> {target_modality}"
    return MBEIR_TASK.get(task_name, None)
def hash_qid(qid):
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id
def hash_did(did):
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id
def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing double quotes may not be necessary
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s

class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"


class MBEIRDatasetBase(Dataset):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
    ):
        """
        Initialize the MBEIRDataset.

        Args:
        - datapath (str): Path to the data.
        - img_preprocess_fn (function): Image preprocessing function.
        - mbeir_data_dir (str): Root directory of the MBEIR dataset.
        - training (bool): Indicator if the dataset is for training.
        """
        self.mbeir_data_dir = mbeir_data_dir

    def __len__(self):
        raise NotImplementedError("This method should be implemented in derived classes.")

    def _load_data_jsonl(self, datapath):
        data_entries = []
        with open(datapath, "r") as fin:
            for line in fin:
                data_entry = json.loads(line)
                data_entries.append(data_entry)
        return data_entries

    def _load_data(self, data_path):
        """Validate and load data."""
        full_data_path = os.path.join(self.mbeir_data_dir, data_path)
        assert os.path.exists(full_data_path), f"Data Path {full_data_path} does not exist"
        assert full_data_path.endswith(".jsonl"), f"Data Path {full_data_path} is not a jsonl file"
        data_entries = self._load_data_jsonl(full_data_path)
        return data_entries

    def _load_query_data(self, query_data_path):
        self.query_data = self._load_data(query_data_path)

    def _load_cand_pool(self, cand_pool_data_path):
        self.cand_pool = self._load_data(cand_pool_data_path)

    def _load_query_instructions(self, instructions_path):
        """Validate and load instructions."""
        full_instructions_path = os.path.join(self.mbeir_data_dir, instructions_path)
        # Validate the path and file extension
        assert os.path.exists(full_instructions_path), f"Instructions Path {full_instructions_path} does not exist"
        assert full_instructions_path.endswith(".tsv"), f"Instructions Path {full_instructions_path} is not a tsv file"
        prompts_dict = {}
        with open(full_instructions_path, "r") as f:
            next(f)  # Skip the header line
            for line in f.readlines():
                parts = line.strip().split("\t")
                # Construct the key to be dataset_id, query_modality, cand_modality
                key = f"{parts[3]}, {parts[0]}, {parts[1]}"
                prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
                prompts_dict[key] = prompts
        self.query_instructions = prompts_dict

    def _load_and_preprocess_image(self, query_img_path):
        """Load an image given a path"""
        if not query_img_path:
            return None
        full_query_img_path = os.path.join(self.mbeir_data_dir, query_img_path)
        assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
        image = Image.open(full_query_img_path)
        # image = self.img_preprocess_fn(image)
        return image

    def _get_random_query_prompt(self, dataset_id, query_modality, cand_modality):
        key = f"{dataset_id}, {query_modality}, {cand_modality}"
        prompts = self.query_instructions.get(key, [])
        assert prompts, f"Cannot find prompts for {key}"
        prompt = format_string(random.choice(prompts))
        assert prompt, f"Prompt is empty for {key}"
        return prompt

    def __getitem__(self, index):
        raise NotImplementedError("This method should be implemented in derived classes.")


class MBEIRMainDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        query_data_path,  # Relate path to the query data
        cand_pool_path,  # Relate path to the candidate pool data
        query_instruct_path,  # Relate path to the query instructions
        mode=Mode.TRAIN,
        enable_query_instruct=True,  # Whether to enable instructions
        shuffle_cand=True,  # Whether to shuffle the candidates
        hard_neg_num=0,  # Number of negative examples in the batch
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(mbeir_data_dir)

        self._load_query_data(query_data_path)
        self._load_cand_pool_as_dict(cand_pool_path)
        self._load_query_instructions(query_instruct_path)

        self.mode = mode
        self.shuffle_cand = shuffle_cand
        self.select_cand = self._get_random_cand if self.shuffle_cand else self._get_first_cand
        self.enable_query_instruct = enable_query_instruct
        self.hard_neg_num = hard_neg_num

        returns = {} if returns is None else returns
        self.returns = {
            "hashed_qid": True,  # default value
            "task_id": True,  # default value
            "hashed_p_did": True,  # default value
            **returns,  # Overwrite defaults with any values provided in returns
        }
        if print_config:
            self.query_data_path = query_data_path
            self.cand_pool_path = cand_pool_path
            self.query_instruct_path = query_instruct_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Mode: {self.mode}")
        print(f"Query Data Path: {self.query_data_path}")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"Enable Query Instructions: {self.enable_query_instruct}")
        if self.enable_query_instruct:
            print(f"Query Instructions Path: {self.query_instruct_path}")
        print(f"Shuffle Candidates: {self.shuffle_cand}")
        print(f"Hard Negative Number: {self.hard_neg_num}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    def _load_cand_pool_as_dict(self, cand_pool_data_path):
        self._load_cand_pool(cand_pool_data_path)
        cand_pool_dict = {}
        for cand_pool_entry in self.cand_pool:
            did = cand_pool_entry.get("did")
            assert did, f"Cannot find did for {cand_pool_entry}"
            cand_pool_dict[did] = cand_pool_entry
        self.cand_pool = cand_pool_dict

    def __len__(self):
        return len(self.query_data)

    def _get_random_cand(self, cand_list):
        return random.choice(cand_list)

    def _get_first_cand(self, cand_list):
        return cand_list[0]

    def __getitem__(self, index):
        """Retrieve an item from the dataset by index."""
        mbeir_entry = self.query_data[index]

        query_txt = mbeir_entry.get("query_txt") or ""
        query_img_path = mbeir_entry.get("query_img_path", None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None

        # Randomly sample a positive example
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        assert len(pos_cand_list) > 0, f"Cannot find positive candidates for {mbeir_entry}"

        # TODO: Fix this hack for OVEN and INFOSEEK
        # We only choose the one matched with the query dataset_id due to OVEN and INFOSEEK
        if self.mode == Mode.EVAL or self.mode == Mode.TEST:
            pos_cand_list = [
                pos_cand_did for pos_cand_did in pos_cand_list if pos_cand_did.split(":")[0] == query_dataset_id
            ]

        selected_pos_cand_did = self.select_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {mbeir_entry}"
        # Note: pos_cand_dataset_id should be the same as query_dataset_id but for OVEN and INFOSEEK it is not.
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        # Randomly sample a query prompt
        # Note:query_modality and pos_cand_modality should define the golden modalities of the current mbeir_entry task.
        # neg_cand_modality could be different from pos_cand_modality.
        query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality)
        # query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_without_prompt = format_string(query_txt)

        # Sample negative examples
        selected_neg_cand_list = []
        if self.mode == Mode.TRAIN:
            neg_cand_id_list = mbeir_entry.get("neg_cand_list", [])
            if self.hard_neg_num > 0:
                assert len(neg_cand_id_list) > 0, f"Cannot find negative candidates for {mbeir_entry}"
                if self.shuffle_cand:
                    random.shuffle(neg_cand_id_list)
                selected_neg_cand_id_list = []
                for i in range(self.hard_neg_num):
                    selected_neg_cand_id_list.append(
                        neg_cand_id_list[i % len(neg_cand_id_list)]
                    )  # % Wrap around from idx 0.
                for neg_cand_did in selected_neg_cand_id_list:
                    neg_cand = self.cand_pool.get(neg_cand_did, None)
                    neg_cand_txt = neg_cand.get("txt") or ""
                    neg_cand_txt = format_string(neg_cand_txt)
                    neg_cand["txt"] = neg_cand_txt
                    selected_neg_cand_list.append(neg_cand)

        def _prepare_data_dict(txt, img_path, modality):
            img = self._load_and_preprocess_image(img_path)
            return {"txt": txt, "img": img, "modality": modality}

        query = _prepare_data_dict(
            query_txt_without_prompt,
            query_img_path,
            query_modality,
        )
        
        instance = {"query": query}
        
        instance.update({"dataset_id": query_dataset_id})
        
        if self.enable_query_instruct:
            instance.update({"prompt": query_prompt})
        else:
            instance.update({"prompt": ""})
            
        if self.returns.get("hashed_p_did"):
            instance.update({"p_did": hash_did(selected_pos_cand_did)})

        pos_cand = _prepare_data_dict(
            pos_cand_txt,
            pos_cand.get("img_path", None),
            pos_cand_modality,
        )
        instance.update({"pos_cand": pos_cand})

        if self.mode == Mode.EVAL or self.mode == Mode.TEST:
            if self.returns.get("hashed_qid"):
                instance.update({"qid": hash_qid(qid)})
            if self.returns.get("task_id"):
                instance.update({"task_id": get_mbeir_task_id(query_modality, pos_cand_modality)})
  
        if self.mode == Mode.TRAIN:
            neg_cand_list = [
                _prepare_data_dict(
                    neg_cand["txt"],
                    neg_cand.get("img_path", None),
                )
                for neg_cand in selected_neg_cand_list
            ]
            if len(neg_cand_list) > 0:
                instance.update({"neg_cand_list": neg_cand_list})
        return instance


class MBEIRCandidatePoolDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        cand_pool_data_path,  # Relate path to the candidate pool data
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(mbeir_data_dir)
        self._load_cand_pool(cand_pool_data_path)

        returns = {} if returns is None else returns
        self.returns = {
            "src_content": False,  # default value
            "hashed_p_did": True,  # default value for candidate id
            **returns,
        }

        # Print dataset config
        if print_config:
            self.cand_pool_path = cand_pool_data_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Candidate Pool Dataset Config---")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    def __len__(self):
        return len(self.cand_pool)

    def __getitem__(self, index):
        mbeir_cand_pool_entry = self.cand_pool[index]
        img_path = mbeir_cand_pool_entry.get("img_path", None)

        did = mbeir_cand_pool_entry.get("did", None)
        dataset_id = did.split(":")[0] if did else None
        cand_txt = mbeir_cand_pool_entry.get("txt") or ""
        cand_txt = format_string(f"{cand_txt}")
        cand_modality = mbeir_cand_pool_entry.get("modality", None)
        
        def _prepare_data_dict(txt, img_path, modality):
            img = self._load_and_preprocess_image(img_path)
            return {"txt": txt, "img": img, "modality": modality}
        instance = {"query" : None}
        
        pos_cand = _prepare_data_dict(
            cand_txt,
            img_path,
            cand_modality,
        )
        
        instance.update({"pos_cand": pos_cand})
        instance.update({"dataset_id": dataset_id})
        instance.update({"prompt": '', 'qid': None, 'task_id': None})
        if self.returns.get("hashed_p_did"):
            instance.update({"p_did": hash_did(did)})
        if self.returns.get("src_content"):
            instance.update({"src_content": mbeir_cand_pool_entry.get("src_content", None)})
        return instance


'''
BEIR Dataset
'''

class BEIRDatasetBase(Dataset):
    def __init__(
        self,
        beir_data_dir,  # Root directory of the MBEIR dataset
    ):
        self.beir_data_dir = beir_data_dir

    def __len__(self):
        raise NotImplementedError("This method should be implemented in derived classes.")

    def _load_data_jsonl(self, datapath):
        data_entries = []
        with open(datapath, "r") as fin:
            for line in fin:
                data_entry = json.loads(line)
                data_entries.append(data_entry)
        return data_entries
    def _load_data_tsv(self, tsv_data_path):
        data_entries = []
        with open(tsv_data_path, "r") as fin:
            for line in fin:
                data_entry = line.strip().split("\t")
                data_entries.append(data_entry)
        return data_entries[1:]  # Skip the header line

    def _load_data(self, data_path):
        """Validate and load data."""
        full_data_path = os.path.join(self.beir_data_dir, data_path)
        assert os.path.exists(full_data_path), f"Data Path {full_data_path} does not exist"
        assert full_data_path.endswith(".jsonl"), f"Data Path {full_data_path} is not a jsonl file"
        data_entries = self._load_data_jsonl(full_data_path)
        return data_entries
    def _load_split_data(self, tsv_data_path):
        full_data_path = os.path.join(self.beir_data_dir, tsv_data_path)
        assert os.path.exists(full_data_path), f"Data Path {full_data_path} does not exist"
        assert full_data_path.endswith(".tsv"), f"Data Path {full_data_path} is not a tsv file"
        data_entries = self._load_data_tsv(full_data_path)
        return data_entries
        

    def _load_query_data(self, query_data_path):
        self.query_data = self._load_data(query_data_path)
    def _load_and_preprocess_image(self, query_img_path):
        """Load an image given a path"""
        if not query_img_path:
            return None
        full_query_img_path = os.path.join(self.mbeir_data_dir, query_img_path)
        assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
        image = Image.open(full_query_img_path)
        # image = self.img_preprocess_fn(image)
        return image
    def _load_cand_pool(self, cand_pool_data_path):
        self.cand_pool = self._load_data(cand_pool_data_path)
    def _load_qrels(self, qrels_data_path):
        self.qrels = self._load_split_data(qrels_data_path)

    def __getitem__(self, index):
        raise NotImplementedError("This method should be implemented in derived classes.")
        '''{
        "query": {
            "txt": "",
            "img": null,
            "modality": ""
        },
        "dataset_id": "",
        "prompt": "",
        "p_did": null,
        "pos_cand": {
            "txt": "",
            "img": null,
            "modality": ""
        }
        }'''

class BEIRMainDataset(BEIRDatasetBase):
    def __init__(
        self,
        beir_data_dir,  # Root directory of the MBEIR dataset
        qrels_data_path,
        query_data_path,  # Relate path to the query data
        cand_pool_path,  # Relate path to the candidate pool data,
        mode=Mode.TRAIN,
        enable_query_instruct=False,  # Whether to enable instructions
        shuffle_cand=True,  # Whether to shuffle the candidates
        hard_neg_num=0,  # Number of negative examples in the batch
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(beir_data_dir)
        
        self._load_query_as_dict(query_data_path)
        self._load_cand_pool_as_dict(cand_pool_path)
        self._load_qrels(qrels_data_path)
        random.shuffle(self.qrels)
        self.mode = mode
        self.shuffle_cand = shuffle_cand
        self.select_cand = self._get_random_cand if self.shuffle_cand else self._get_first_cand
        self.enable_query_instruct = enable_query_instruct
        self.hard_neg_num = hard_neg_num

        returns = {} if returns is None else returns
        self.returns = {
            "hashed_qid": True,  # default value
            "task_id": True,  # default value
            "hashed_p_did": True,  # default value
            **returns,  # Overwrite defaults with any values provided in returns
        }
        if print_config:
            self.query_data_path = query_data_path
            self.cand_pool_path = cand_pool_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Mode: {self.mode}")
        print(f"Query Data Path: {self.query_data_path}")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"Enable Query Instructions: {self.enable_query_instruct}")
        if self.enable_query_instruct:
            print(f"Query Instructions Path: {self.query_instruct_path}")
        print(f"Shuffle Candidates: {self.shuffle_cand}")
        print(f"Hard Negative Number: {self.hard_neg_num}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    def _load_cand_pool_as_dict(self, cand_pool_data_path):
        self._load_cand_pool(cand_pool_data_path)
        cand_pool_dict = {}
        for cand_pool_entry in self.cand_pool:
            did = cand_pool_entry.get("_id")
            assert did, f"Cannot find did for {cand_pool_entry}"
            cand_pool_dict[did] = cand_pool_entry
        self.cand_pool = cand_pool_dict
    def _load_query_as_dict(self, cand_pool_data_path):
        self._load_query_data(cand_pool_data_path)
        query_data_dict = {}
        for cand_pool_entry in self.query_data:
            did = cand_pool_entry.get("_id")
            assert did, f"Cannot find did for {cand_pool_entry}"
            query_data_dict[did] = cand_pool_entry
        self.query_data = query_data_dict

    def __len__(self):
        return len(self.qrels)

    def _get_random_cand(self, cand_list):
        return random.choice(cand_list)

    def _get_first_cand(self, cand_list):
        return cand_list[0]

    def __getitem__(self, index):
        """Retrieve an item from the dataset by index."""
        def _get_beir_entry(qrel, query_data):
            q_id, pos_cand_id = qrel[0], qrel[1]
            beir_entry = {
                        "qid": q_id,
                        "query_txt": query_data[q_id].get("text"),
                        "query_img_path": "",
                        "query_modality": "text",
                        "pos_cand_list": [
                            pos_cand_id
                        ],
                        "neg_cand_list": [],
                        "task_id": 1
                        }
            return beir_entry
        beir_entry = _get_beir_entry(self.qrels[index], self.query_data)
        query_txt = beir_entry.get("query_txt") or ""
        query_img_path = beir_entry.get("query_img_path", None)
        query_modality = beir_entry.get("query_modality", None)
        qid = beir_entry.get("qid", None)

        # Randomly sample a positive example
        pos_cand_list = beir_entry.get("pos_cand_list", [])
        assert len(pos_cand_list) > 0, f"Cannot find positive candidates for {beir_entry}"

        selected_pos_cand_did = self.select_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {beir_entry}"

        pos_cand_modality = pos_cand.get("modality", "text")
        pos_cand_txt = pos_cand.get("text") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        query_txt_without_prompt = format_string(query_txt)
        query_prompt = ""
        # Sample negative examples
        selected_neg_cand_list = []
        if self.mode == Mode.TRAIN:
            neg_cand_id_list = beir_entry.get("neg_cand_list", [])
            if self.hard_neg_num > 0:
                assert len(neg_cand_id_list) > 0, f"Cannot find negative candidates for {beir_entry}"
                if self.shuffle_cand:
                    random.shuffle(neg_cand_id_list)
                selected_neg_cand_id_list = []
                for i in range(self.hard_neg_num):
                    selected_neg_cand_id_list.append(
                        neg_cand_id_list[i % len(neg_cand_id_list)]
                    )  # % Wrap around from idx 0.
                for neg_cand_did in selected_neg_cand_id_list:
                    neg_cand = self.cand_pool.get(neg_cand_did, None)
                    neg_cand_txt = neg_cand.get("txt") or ""
                    neg_cand_txt = format_string(neg_cand_txt)
                    neg_cand["txt"] = neg_cand_txt
                    selected_neg_cand_list.append(neg_cand)

        def _prepare_data_dict(txt, img_path, modality):
            img = self._load_and_preprocess_image(img_path)
            return {"txt": txt, "img": img, "modality": modality}

        query = _prepare_data_dict(
            query_txt_without_prompt,
            query_img_path,
            query_modality,
        )
        
        instance = {"query": query}
        
        if self.enable_query_instruct:
            instance.update({"prompt": query_prompt})
        else:
            instance.update({"prompt": ""})
            
        if self.returns.get("hashed_p_did"):
            instance.update({"p_did": selected_pos_cand_did})

        pos_cand = _prepare_data_dict(
            pos_cand_txt,
            pos_cand.get("img_path", None),
            pos_cand_modality,
        )
        instance.update({"pos_cand": pos_cand})

        if self.mode == Mode.EVAL or self.mode == Mode.TEST:
            if self.returns.get("hashed_qid"):
                instance.update({"qid": qid})
            if self.returns.get("task_id"):
                instance.update({"task_id": get_mbeir_task_id(query_modality, pos_cand_modality)})
  
        if self.mode == Mode.TRAIN:
            neg_cand_list = [
                _prepare_data_dict(
                    neg_cand["txt"],
                    neg_cand.get("img_path", None),
                )
                for neg_cand in selected_neg_cand_list
            ]
            if len(neg_cand_list) > 0:
                instance.update({"neg_cand_list": neg_cand_list})
        return instance


class BEIRCandidatePoolDataset(BEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        cand_pool_data_path,  # Relate path to the candidate pool data
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(mbeir_data_dir)
        self._load_cand_pool(cand_pool_data_path)

        returns = {} if returns is None else returns
        self.returns = {
            "src_content": False,  # default value
            "hashed_p_did": True,  # default value for candidate id
            **returns,
        }

        # Print dataset config
        if print_config:
            self.cand_pool_path = cand_pool_data_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Candidate Pool Dataset Config---")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")
    def __len__(self):
        return len(self.cand_pool)

    def __getitem__(self, index):
        '''
            {
            "query": null,
            "pos_cand": {
                "txt": "",
                "img": null,
                "modality": ""
            },
            "dataset_id": "",
            "prompt": "",
            "qid": null,
            "task_id": null,
            "p_did": null
            }
        '''
        mbeir_cand_pool_entry = self.cand_pool[index]
        img_path = mbeir_cand_pool_entry.get("img_path", None)

        did = mbeir_cand_pool_entry.get("_id", None)
        cand_txt = mbeir_cand_pool_entry.get("text") or ""
        cand_txt = format_string(f"{cand_txt}")
        cand_modality = mbeir_cand_pool_entry.get("modality", "text")
        def _prepare_data_dict(txt, img_path, modality):
            img = self._load_and_preprocess_image(img_path)
            return {"txt": txt, "img": img, "modality": modality}
        instance = {"query" : None}
        
        pos_cand = _prepare_data_dict(
            cand_txt,
            img_path,
            cand_modality,
        )
        
        instance.update({"pos_cand": pos_cand})
        instance.update({"prompt": '', 'qid': None, 'task_id': None})
        if self.returns.get("hashed_p_did"):
            instance.update({"p_did": did})
        if self.returns.get("src_content"):
            instance.update({"src_content": mbeir_cand_pool_entry.get("src_content", None)})
        return instance
