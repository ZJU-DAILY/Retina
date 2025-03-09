import torch
from transformers import Trainer


class RerankerTrainer(Trainer):
    def __init__(self, *args, loss_func,  **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        label = inputs.pop("combined_label")
        logits = model(**{k[9:]: v for k, v in inputs.items() if k.startswith("combined")})
        
        return self.loss_func(logits, label)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")
        with torch.no_grad():
            label = inputs.pop("combined_label")
            logits = model(**{k[9:]: v for k, v in inputs.items() if k.startswith("combined")})
            loss = self.loss_func(logits, label)
            
        return loss, None, None
