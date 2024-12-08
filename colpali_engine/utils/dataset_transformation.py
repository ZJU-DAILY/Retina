import os
from typing import List, Tuple, cast
from .mbeir_dataset import MBEIRMainDataset, MBEIRInferenceOnlyDataset, MBEIRCandidatePoolDataset, Mode
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "1") == "1"

def add_metadata_column(dataset, column_name, value):
    def add_source(example):
        example[column_name] = value
        return example

    return dataset.map(add_source)

def load_train_set() -> DatasetDict:
    print("load_train_set")
    ds_path = "colpali_train_set"
    base_path =  '/data2/zhh_data/' if USE_LOCAL_DATASET else "vidore/"
    dataset = cast(DatasetDict, load_dataset(base_path + ds_path))['train']
    
    dataset_eval = dataset.select(range(50))
    dataset = dataset.select(range(50, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
        
    return ds_dict


def load_icrr_train_set(        
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        query_instruct_path,  # Relate path to the query data
        train_query_data_path,  # Relate path to the candidate pool data
        train_cand_pool_data_path,
        val_query_data_path,
        val_cand_pool_data_path
        ) -> DatasetDict:
    
    train_dataset = MBEIRMainDataset(        
                    mbeir_data_dir,  
                    train_query_data_path,  
                    train_cand_pool_data_path,
                    query_instruct_path,
                    mode=Mode.TRAIN)
    val_dataset = MBEIRMainDataset(
                    mbeir_data_dir,    
                    val_query_data_path,  
                    val_cand_pool_data_path,
                    query_instruct_path,
                    mode=Mode.EVAL)
    
    ds_dict = DatasetDict({"train": train_dataset, "val": val_dataset})
        
    return ds_dict


def load_icrr_test_set(       
        mbeir_data_dir,  
        query_instruct_path,  
        test_query_data_path,
        test_cand_pool_data_path,
        ) -> DatasetDict:
    
    test_dataset = MBEIRMainDataset(
                    mbeir_data_dir,  
                    test_query_data_path, 
                    test_cand_pool_data_path,
                    query_instruct_path,
                    mode=Mode.TEST)
    
    candidatepool_dataset = MBEIRCandidatePoolDataset(
                    mbeir_data_dir,
                    test_cand_pool_data_path)
    return test_dataset, candidatepool_dataset


def load_train_set_detailed() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_train_set_with_tabfquad() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docmatix_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "docmatix-ir", split="train"))
    # dataset = dataset.select(range(100500))

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "Docmatix", "images", split="train"))

    return ds_dict, anchor_ds, "docmatix"

def load_wikiss() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "wiki-ss-nq", data_files="train.jsonl", split="train"))
    # dataset = dataset.select(range(400500))
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "wiki-ss-corpus", split="train"))

    return ds_dict, anchor_ds, "wikiss"


def load_train_set_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "manu/"
    dataset = cast(Dataset, load_dataset(base_path + "colpali-queries", split="train"))

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    print("Dataset size after filtering:", len(dataset))

    # keep only top 20 negative passages
    dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:20]})

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    anchor_ds = cast(Dataset, load_dataset(base_path + "colpali-corpus", split="train"))
    return ds_dict, anchor_ds, "vidore"


def load_train_set_with_docmatix() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
        "Docmatix_filtered_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot: List[Dataset] = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = concatenate_datasets(ds_tot)
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docvqa_dataset() -> DatasetDict:
    if USE_LOCAL_DATASET:
        dataset_doc = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="test"))
    else:
        dataset_doc = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test"))

    # concatenate the two datasets
    dataset = concatenate_datasets([dataset_doc, dataset_info])
    dataset_eval = concatenate_datasets([dataset_doc_eval, dataset_info_eval])
    # sample 100 from eval dataset
    dataset_eval = dataset_eval.shuffle(seed=42).select(range(200))

    # rename question as query
    dataset = dataset.rename_column("question", "query")
    dataset_eval = dataset_eval.rename_column("question", "query")

    # create new column image_filename that corresponds to ucsf_document_id if not None, else image_url
    dataset = dataset.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )
    dataset_eval = dataset_eval.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )

    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    return ds_dict   


class TestSetFactory:
    def __init__(self, 
                 mbeir_data_dir = '',
                 test_query_data_path = '', 
                 test_cand_pool_data_path = ''):
        self.test_query_data_path = os.path.join('query/test', test_query_data_path)
        self.test_cand_pool_data_path = os.path.join('cand_pool/local', test_cand_pool_data_path)
        self.mbeir_data_dir = mbeir_data_dir

    def __call__(self, *args, **kwargs):
        
        # dataset = load_icrr_test_set("/share/home/22351087/M-BEIR/", 
        #                              "instructions/query_instructions.tsv",
        #                              self.test_query_data_path,
        #                              self.test_cand_pool_data_path)
        
        dataset = load_icrr_test_set(self.mbeir_data_dir, 
                                "instructions/query_instructions.tsv",
                                self.test_query_data_path,
                                self.test_cand_pool_data_path)
        return dataset


if __name__ == "__main__":
    ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
    print(ds)
