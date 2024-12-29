import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import copy
import torch
from datasets import concatenate_datasets, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

from colpali_engine.collators import CorpusQueryCollator, VisualRetrieverCollator
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
)
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.trainer.eval_utils import CustomRetrievalEvaluator, score_processing
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@dataclass
class ColModelTrainingConfig:
    model: PreTrainedModel
    tr_args: TrainingArguments = None
    output_dir: str = None
    max_length: int = 256
    use_example: bool = False
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    processor: BaseVisualRetrieverProcessor = None
    tokenizer: PreTrainedTokenizer = None
    loss_func: Optional[Callable] = ColbertLoss()
    dataset_loading_func: Optional[Callable] = None
    eval_dataset_loader: Optional[Dict[str, Callable]] = None
    pretrained_peft_model_name_or_path: Optional[str] = None

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None:
            self.tr_args.output_dir = self.output_dir

        # cast if string
        if isinstance(self.tr_args.learning_rate, str):
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)
        self.tr_args.remove_unused_columns = False

        if self.processor is None and self.tokenizer is None:
            print("Using textual model tokenization")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)

        if self.pretrained_peft_model_name_or_path is not None:
            self.model.load_adapter(self.pretrained_peft_model_name_or_path)
            print(f"Loaded pretrained adapter from {self.pretrained_peft_model_name_or_path}")

        if self.peft_config is not None:
            print("Configurating PEFT model")
            if self.processor is None:
                # Might be deprecated - use the "else" branch
                self.model = prepare_model_for_kbit_training(self.model)  # use_gradient_checkpointing=True
                # self.model.enable_input_require_grads()
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
            else:
                if self.pretrained_peft_model_name_or_path is None:
                    # self.model.add_adapter(self.peft_config)
                    # self.model.enable_adapters()
                    self.model = get_peft_model(self.model, self.peft_config)
                    self.model.print_trainable_parameters()
                else:
                    print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")
    print_gpu_utilization()


class ColModelTraining:
    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model

        self.dataset =self.config.dataset_loading_func

        if isinstance(self.dataset, Tuple):
            corpus_format = self.dataset[2]
            neg_dataset = self.dataset[1]
            self.dataset = self.dataset[0]
            self.collator = CorpusQueryCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
                image_dataset=neg_dataset,
                mined_negatives=True,
                corpus_format=corpus_format,
            )
        else:
            self.collator = VisualRetrieverCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
                use_example=self.config.use_example
            )
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.retrieval_evaluator = CustomRetrievalEvaluator()

    def train(self) -> None:
        if isinstance(self.collator, CorpusQueryCollator) and self.collator.mined_negatives:
            print("Training with hard negatives")
        else:
            print("Training with in-batch negatives")

        trainer = ContrastiveTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["val"],
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
        )

        trainer.args.remove_unused_columns = False

        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

    def eval_dataset(self, test_dataset, candidatepool_dataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if next(self.model.parameters()).is_cuda:
            print("Model is already on GPU.")
        else:
            self.model = self.model.to(device)
            print("Model is not on GPU. Moving to:", device)
        self.model.eval()
        
        # debug
        # test_dataset = Subset(test_dataset, range(150))
        # candidatepool_dataset = Subset(candidatepool_dataset, range(250))
        # debug
        
        dataloader_with_query = DataLoader(
            test_dataset,
            batch_size=self.config.tr_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )
        dataloader_without_query = DataLoader(
            candidatepool_dataset,
            batch_size=self.config.tr_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )
            
        relevant_docs = {}
        docidx_2_docid = {}
        qsidx_2_query = []
        for idx, sample in enumerate(test_dataset):
            doc_id = str(sample["p_did"])
            if sample["query"] is not None:
                relevant_docs[str(idx)] = {doc_id: 1}
                qsidx_2_query.append(str(idx))

        for idx, sample in enumerate(candidatepool_dataset):
            doc_id = str(sample["p_did"])
            docidx_2_docid[str(idx)] = doc_id
        
        if os.path.exists(f"{self.config.output_dir}/qs.pt") and os.path.exists(f"{self.config.output_dir}/ps.pt"):
            qs = torch.load(f"{self.config.output_dir}/qs.pt", weights_only=True)
            ps = torch.load(f"{self.config.output_dir}/ps.pt", weights_only=True)
            # ps = torch.load(f"{self.config.output_dir}/qs.pt", weights_only=True)
            
            print("Embeddings already computed, loading")
        else:
            qs = []
            ps = []
            device = self.model.device
            with torch.no_grad():
                for dataloader in [dataloader_with_query, dataloader_without_query]:
                    for batch in tqdm(dataloader):
                        if "query_input_ids" in batch:
                            query = self.model(**{k[6:]: v.to(device) for k, v in batch.items() if k.startswith("query")})
                            qs.extend(list(torch.unbind(query.to("cpu"))))
                        else:
                            doc = self.model(**{k[4:]: v.to(device) for k, v in batch.items() if k.startswith("doc")})
                            ps.extend(list(torch.unbind(doc.to("cpu"))))

            print("Embeddings computed, evaluating")
            #save embeddings
            torch.save(qs, f"{self.config.output_dir}/qs.pt")
            torch.save(ps, f"{self.config.output_dir}/ps.pt")
        
        scores = self.config.processor.score(qs, ps, device=self.model.device)
        results = {}
        assert scores.shape[0] == len(qsidx_2_query)
        for idx, scores_per_query in enumerate(scores):
            results[qsidx_2_query[idx]] = {
                docidx_2_docid[str(docidx)]: float(score) for docidx, score in enumerate(scores_per_query)
            }
        # results = score_processing(scores, qsidx_2_query, docidx_2_docid)
        del scores
        # evaluate
        metrics = self.retrieval_evaluator.compute_mteb_metrics(relevant_docs, results)
        print("MTEB metrics:", metrics)
        
        # delete embeddings
        os.remove(f"{self.config.output_dir}/qs.pt")
        os.remove(f"{self.config.output_dir}/ps.pt")

        return metrics

    def eval(self) -> None:
        all_metrics = {}
        
        # try:
        #     print("\nEvaluating on validation set")
        #     metrics = self.eval_dataset(self.dataset["val"])
        #     print(f"\nMetrics for validation set: {metrics}")
        #     all_metrics["validation_set"] = metrics
        # except Exception as e:
        #     print(f"Error evaluating validation set: {e}")
        # switching to normal collator
        
        self.collator = VisualRetrieverCollator(
            processor=self.config.processor,
            max_length=self.config.max_length,
        )
        if self.config.eval_dataset_loader is not None:
            for test_name, test_dataset_loading_func in self.config.eval_dataset_loader.items():
                print(f"\nEvaluating {test_name}")
                test_ds, cand_ds = test_dataset_loading_func()
                metrics = self.eval_dataset(test_ds, cand_ds)
                all_metrics[test_name] = metrics
                print(f"\nMetrics for {test_name}: {metrics}")

                # checkpoint dumps
                with open(f"{self.config.output_dir}/results_use_example_{self.config.use_example}.json", "a") as f:
                    json.dump(all_metrics, f)
                    
        # save results as json
        with open(f"{self.config.output_dir}/results_use_example_{self.config.use_example}.json", "a") as f:
            json.dump(all_metrics, f)

    def save(self, config_file):
        # save model
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(self.config.output_dir)
        # self.model.base_model.save_pretrained(os.path.join(self.config.output_dir, "base_model"))
        if self.config.tokenizer is not None:
            self.config.tokenizer.save_pretrained(self.config.output_dir)
        if self.config.processor is not None:
            self.config.processor.save_pretrained(self.config.output_dir)  # save config

        # copy-paste the yml file with os
        os.system(f"cp {config_file} {self.config.output_dir}/training_config.yml")

        # save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
