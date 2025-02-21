import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss


class FLOPS(nn.Module):
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """
    def __init__(self):
        super().__init__()
    def forward(self, embeddings):
        return torch.sum(torch.mean(torch.abs(embeddings), dim=0) ** 2)

class SparseEncoderLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)
        loss_rowwise = self.ce_loss(scores, torch.arange(scores.shape[0], device=scores.device))

        return loss_rowwise

class SparsePairwiseCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

        pos_scores = scores.diagonal()
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6
        neg_scores = neg_scores.max(dim=1)[0]

        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss

class SparsePairwiseFlopsLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.flops_loss = FLOPS()
        self.lambda_flops_query = 6e-5
        self.lambda_flops_doc = 2e-5
    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """
        
        print("query_embeddings", query_embeddings.shape, torch.sum(torch.sum(query_embeddings > 0, dim=0).float()) / query_embeddings.size(0))
        print("doc_embeddings", doc_embeddings.shape, torch.sum(torch.sum(doc_embeddings > 0, dim=0).float()) / doc_embeddings.size(0))

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

        pos_scores = scores.diagonal()
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6
        neg_scores = neg_scores.max(dim=1)[0]

        loss_ce = F.softplus(neg_scores - pos_scores).mean()
        loss_flops_query = self.flops_loss(query_embeddings)
        loss_flops_doc = self.flops_loss(doc_embeddings)
        loss = loss_ce + self.lambda_flops_query * loss_flops_query + self.lambda_flops_doc * loss_flops_doc

        return loss


class SparsePairwiseNegativeCELoss(torch.nn.Module):
    def __init__(self, in_batch_term=False):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.in_batch_term = in_batch_term

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        neg_doc_embeddings: (batch_size, dim)
        """
        
        pos_scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings).diagonal()
        neg_scores = torch.einsum("bd,cd->bc", query_embeddings, neg_doc_embeddings).diagonal()

        loss = F.softplus(neg_scores - pos_scores).mean()

        if self.in_batch_term:
            scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)

            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            loss += F.softplus(neg_scores - pos_scores).mean()

        return loss / 2
