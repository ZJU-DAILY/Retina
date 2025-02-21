import os
from typing import List, Tuple, cast
from .mbeir_dataset import MBEIRMainDataset, MBEIRCandidatePoolDataset, Mode
from .beir_dataset import BEIRMainDataset, BEIRCandidatePoolDataset
from .triple_dataset import TripleCandidatePoolDataset,TripleTrainDataset, TripleTestDataset
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import  ConcatDataset

# USE_BASE_PATH = os.environ.get("USE_BASE_PATH", 0)
USE_BASE_PATH = "/data2/zhh_data/"

def add_metadata_column(dataset, column_name, value):
    def add_source(example):
        example[column_name] = value
        return example

    return dataset.map(add_source)

def load_train_set_detailed() -> DatasetDict:
    
    ds_paths = [
        "test",
    ]
    if not USE_BASE_PATH:
        raise ValueError("USE_BASE_PATH must be set to a valid path") 
    base_path = USE_BASE_PATH
    ds_tot = []
    for path in ds_paths:
        data_dir = os.path.join(base_path,"VisualIR",path)
        ds = load_dataset(data_dir)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    # split into train and test
    dataset_eval = dataset.select(range(100))
    dataset = dataset.select(range(100, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "eval": dataset_eval})
    return ds_dict


def load_mbeir_train_set(
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


def load_mbeir_test_set(       
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



def load_beir_train_set(        
        beir_data_dir,  # Root directory of the MBEIR dataset
        train_qrels_data_path,  
        dev_qrels_data_path,
        query_data_path,
        cand_pool_data_path,
        ) -> DatasetDict: 
    
    train_dataset = BEIRMainDataset(        
                    beir_data_dir,  
                    train_qrels_data_path,
                    query_data_path,
                    cand_pool_data_path,
                    mode=Mode.TRAIN)
    val_dataset = BEIRMainDataset(
                    beir_data_dir,
                    dev_qrels_data_path,
                    query_data_path,
                    cand_pool_data_path,
                    mode=Mode.EVAL)
    ds_dict = DatasetDict({"train": train_dataset, "val": val_dataset})
        
    return ds_dict

def load_beir_test_set(
        beir_data_dir, 
        test_qrels_data_path,  
        query_data_path,
        cand_pool_data_path,
        ) -> DatasetDict:
    
    test_dataset = BEIRMainDataset(
                    beir_data_dir, 
                    test_qrels_data_path,
                    query_data_path, 
                    cand_pool_data_path,
                    mode=Mode.TEST)
    
    candidatepool_dataset = BEIRCandidatePoolDataset(
                    beir_data_dir,
                    cand_pool_data_path)
    return test_dataset, candidatepool_dataset


class TestSetFactory:
    def __init__(self, 
                 mbeir_data_dir = '',
                 test_query_data_path = '', 
                 test_cand_pool_data_path = ''):
        self.test_query_data_path, = os.path.join('query/test', test_query_data_path),
        self.test_cand_pool_data_path,  = os.path.join('cand_pool/local', test_cand_pool_data_path),
        self.mbeir_data_dir = mbeir_data_dir

    def __call__(self, *args, **kwargs):
        
        # dataset = load_icrr_test_set("/share/home/22351087/M-BEIR/", 
        #                              "instructions/query_instructions.tsv",
        #                              self.test_query_data_path,
        #                              self.test_cand_pool_data_path)
        
        dataset = load_mbeir_test_set(self.mbeir_data_dir, 
                                "instructions/query_instructions.tsv",
                                self.test_query_data_path,
                                self.test_cand_pool_data_path)
        return dataset
    
class BeirTestSetFactory:
    def __init__(self, 
                 data_dir = ''):
        self.data_dir = os.path.join(USE_BASE_PATH, data_dir)

    def __call__(self, *args, **kwargs):
        
        test_dataset = load_dataset(self.data_dir, name = 'queries', split='train')
        candidatepool_dataset = load_dataset(self.data_dir, name = 'corpus', split='train')
        
        return test_dataset, candidatepool_dataset

if __name__ == "__main__":
    ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
    print(ds)
