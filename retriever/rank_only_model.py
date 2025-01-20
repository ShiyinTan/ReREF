import math
from typing import Type
import numpy as np
import pytorch_lightning as pl

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    AutoModel,
    AutoConfig,
    LEDModel,
    LEDForConditionalGeneration,
    LEDConfig,
    Adafactor
)
from utils import SelfAttentiveSpanExtractor, get_spans_embeddings_with_index_select, \
    search_doc_edu_id, get_rank_from_rank_multi
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pdb
# from datasets import load_dataset, load_metric
from datasets import load_dataset
import evaluate
import os
import torch.distributed as dist
from itertools import chain
from transformers.utils import logging
from collections import defaultdict
logging.set_verbosity_error() 



class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dim_decrease_ratio=8, dropout: float = 0.0):
        """
        Multi-Layer Perceptron Classifier.
        :param input_dim: int, dimension of input
        :param dropout: float, dropout rate
        """
        super().__init__()
        hid_dim_1 = input_dim//dim_decrease_ratio
        hid_dim_2 = hid_dim_1//dim_decrease_ratio
        self.fc1 = nn.Linear(input_dim, hid_dim_1)
        self.fc2 = nn.Linear(hid_dim_1, hid_dim_2)
        self.fc3 = nn.Linear(hid_dim_2, 1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(hid_dim_1)
        self.norm_2 = nn.LayerNorm(hid_dim_2)

    def forward(self, x: torch.Tensor):
        """
        multi-layer perceptron classifier forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """

        x = self.dropout(self.act(self.fc1(x)))

        x = self.dropout(self.act(self.fc2(x)))
        return self.fc3(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class GatedPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, temp=1.0, sparse=False, top_k=16) -> torch.Tensor:
        """
        x: [B, L, D], tokens of length L
        temperature for softmax
        sparse pooling: select top_k token from x to pooling 
        return: 
            pooled_embeddings: [B, D]
            routing_weights: [B, L, 1]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        hidden_states = self.fc(x)
        router_logits = self.gate(x)
        if torch.any(torch.isnan(router_logits)):
            print("Input contains NaN values")
        if torch.any(torch.isinf(router_logits)):
            print("Input contains infinite values")
        router_logits_stable = router_logits - torch.max(router_logits) # to avoid nan
        routing_weights = F.softmax(router_logits_stable/temp, dim=1, dtype=torch.float)
        if sparse:
            routing_weights, selected_embeddings = torch.topk(routing_weights, top_k, dim=1)
            routing_weights /= routing_weights.sum(dim=1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)
        
        pooled_embeddings = (hidden_states * routing_weights).sum(1)
        return pooled_embeddings, routing_weights



class MultiheadAttentionLayer(nn.Module):
    def __init__(self, input_dim, hid_dim_qk, hid_dim_v, n_heads, dropout=0):
        """
        hid_dim_qk: hidden dimension of query and key should be equal
        """
        super().__init__()
        assert hid_dim_qk % n_heads == 0
        assert hid_dim_v % n_heads == 0
        self.hid_dim_qk = hid_dim_qk
        self.hid_dim_v = hid_dim_v
        self.n_heads = n_heads
        self.head_dim_qk = hid_dim_qk//n_heads
        self.head_dim_v = hid_dim_v//n_heads
        
        self.fc_q = nn.Linear(input_dim, hid_dim_qk)
        self.fc_k = nn.Linear(input_dim, hid_dim_qk)
        self.fc_v = nn.Linear(input_dim, hid_dim_v)
        
        self.fc_o = nn.Linear(hid_dim_v, hid_dim_v)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.hid_dim_qk) # normalized by dim of qk
    
    def forward(self, query, key, value, temp=1.0, mask = None):
        """
        key_len should equal to value_len
        @input: 
            query = [batch_size, query_len, input_dim]
            key = [batch_size, key_len, input_dim]
            value = [batch_size, value_len, input_dim]
            temp: temperature
        @output: 
            x: [batch, query_len, hid_dim_v]
            attention: [batch, n_head, query_len, value_len]
        """
        
        batch_size = query.shape[0]
        
        Q = self.fc_q(query) # [bs, query len, hid dim_q]
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim_qk).permute(0,2,1,3) # [bs, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim_qk).permute(0,2,1,3) # [bs, n heads, key len, head dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim_v).permute(0,2,1,3) # [bs, n heads, value len, head dim]
        
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale # [bs, n heads, query len, key len]
        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)
        
        attention = torch.softmax(energy/temp, dim=-1) # [bs, n heads, query len, key len]
        x = torch.matmul(self.dropout(attention), V) # [bs, n heads, query len, head dim]
        x = x.permute(0,2,1,3).contiguous() # [bs, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim_v) # [bs, query len, hid dim]
        x = self.fc_o(x)
        return x, attention


def edu_rank_loss(predicted_scores, ground_truth_labels):
    return 0


def edu_bpr_loss(predicted_scores, ground_truth_labels, top_percentile=10, bottom_percentile=20, topk_middle=5):
    """
    the difference between edu_bpr and bpr is the positive and negative samples selection
    bpr: positive sample = one true position, negative samples = all docs - positive sample
    edu_bpr: 
    for query selection:
            top 20% positions are true positions, and other 80% positions are negative positions
    for filtering:
            top 80% positions are true positions, and bottom 20% positions are negative positions
    But we don't care the ranks between top 20% and botton 20%, therefore, we only use top 20% as positives
    and bottom 20% as negativs, so we simplify the bpr loss
    Furthermore, only rank top 20% before than bottom 20% is not enough, since top 20% may not rank the topest,
    and bottom 20% may not rank bottomest, we random sample topk_middle samples as positive samples for filterng, 
    and sample topk_middle samples as negatives samples for query selections
    """
    batch_size, num_items = predicted_scores.size()
    predicted_scores = predicted_scores.squeeze(0)
    ground_truth_labels = ground_truth_labels.squeeze(0)
    top_percentile = top_percentile/100
    bottom_percentile = bottom_percentile/100
    top_percentile_position = max(1, int(num_items*(top_percentile)))
    if top_percentile_position < 10:
        return 0
    else:
        top_percentile_position = min(top_percentile_position, 10)
    bottom_percentile_position = min(num_items-2, int(num_items*(1-bottom_percentile)))
    
    positive_positions = ground_truth_labels[:top_percentile_position]
    negative_positions = ground_truth_labels[bottom_percentile_position:]

    positives = predicted_scores[positive_positions]
    negatives = predicted_scores[negative_positions]

    distances_matrix = positives.unsqueeze(1) - negatives.unsqueeze(0)
    
    if top_percentile_position >= bottom_percentile_position:
        loss_positives = 0
        loss_negatives = 0
    else:
        middle_positions = ground_truth_labels[top_percentile_position:bottom_percentile_position]
        middle_positives = predicted_scores[torch.randperm(len(middle_positions))[:topk_middle]]
        middle_negatives = predicted_scores[torch.randperm(len(middle_positions))[:topk_middle]]
        
        distances_matrix_positives = positives.unsqueeze(1) - middle_negatives.unsqueeze(0)
        distances_matrix_negatives = middle_positives.unsqueeze(1) - negatives.unsqueeze(0)
        
        loss_positives = -torch.mean(torch.log(torch.sigmoid(distances_matrix_positives)+1e-6))
        loss_negatives = -torch.mean(torch.log(torch.sigmoid(distances_matrix_negatives)+1e-6))
    
    
    loss = -torch.mean(torch.log(torch.sigmoid(distances_matrix)+1e-6))
    if np.isnan(loss_positives.cpu().detach().numpy()) \
        or np.isnan(loss_negatives.cpu().detach().numpy()) \
        or np.isnan(loss.cpu().detach().numpy()):
        return 0
    return loss + loss_positives + loss_negatives

def multi_query_listmle_loss(predicted_scores_multi_query, ground_truth_labels):
    """
    predicted_scores: Tensor of shape [num_docs, num_query]
    ground_truth_labels: Tensor of shape [batch_size, num_items], batch_size=1 default
    """
    loss = 0
    query_num = predicted_scores_multi_query.shape[1]
    for query_id in range(predicted_scores_multi_query.shape[1]):
        predicted_scores = predicted_scores_multi_query[:,query_id].unsqueeze(0)
        batch_size, num_items = predicted_scores.size()

        loss += listmle_loss(predicted_scores, ground_truth_labels)

    return loss/query_num


def multi_query_bpr_loss(predicted_scores_multi_query, ground_truth_labels):
    """
    predicted_scores: Tensor of shape [num_docs, num_query]
    ground_truth_labels: Tensor of shape [batch_size, num_items], batch_size=1 default
    """
    loss = 0
    query_num = predicted_scores_multi_query.shape[1]
    if query_num == 0:
        return loss
    for query_id in range(predicted_scores_multi_query.shape[1]):
        predicted_scores = predicted_scores_multi_query[:,query_id].unsqueeze(0)
        batch_size, num_items = predicted_scores.size()

        loss += bpr_loss(predicted_scores, ground_truth_labels)
        
    return loss/query_num



def bpr_loss(predicted_scores, ground_truth_labels, topk=5):
    """
    ensure topk rank, we only calculate topk positive positions, 
    which means only topk important documents is necessary
    If all documents is used for rank, the last document is not need to calculate loss, 
    since last document has no negative samples, all documents are positive samples.
    predicted_scores: Tensor of shape [batch_size, num_items]
    ground_truth_labels: Tensor of shape [batch_size, num_items]
    """
    
    batch_size, num_items = predicted_scores.size()
    predicted_scores = predicted_scores.squeeze(0)
    ground_truth_labels = ground_truth_labels.squeeze(0)
    topk = min(num_items-1, topk)
    loss = 0.0
    
    for i in range(topk):
        positive_position = ground_truth_labels[i]
        negative_positions = ground_truth_labels[(i+1):]
        positive = predicted_scores[positive_position]
        negatives = predicted_scores[negative_positions]
        
        distance = positive - negatives
        loss += -torch.mean(torch.log(torch.sigmoid(distance)+1e-6))

    return loss


def listmle_loss(predicted_scores, ground_truth_labels):
    """
    predicted_scores: Tensor of shape [batch_size, num_items]
    ground_truth_labels: Tensor of shape [batch_size, num_items]
    """
    batch_size, num_items = predicted_scores.size()
    # Get the descending sort indices based on ground truth labels
    _, sorted_indices = torch.sort(ground_truth_labels, descending=True, dim=1)
    # Apply the permutation to predicted scores
    sorted_scores = torch.gather(predicted_scores, 1, sorted_indices)
    # Compute the loss
    loss = 0.0
    for i in range(num_items):
        # For each position, compute the denominator
        denominators = torch.logsumexp(sorted_scores[:, i:], dim=1)
        # Subtract the score at position i
        loss += denominators - sorted_scores[:, i]
    # Take the mean over the batch
    loss = torch.mean(loss)
    return loss

def ranking_loss(score, rank, margin=0.01):
    # equivalent to initializing TotalLoss to 0
    # here is to avoid that some special samples will not go into the following for loop
    ones = torch.ones(score.size()).cuda(score.device)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)

    # candidate loss
    n = score.size(1)
    for i in range(1, n):
        pos_score = score[:, :-i]
        neg_score = score[:, i:]
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(margin * i)
        TotalLoss += loss_func(pos_score, neg_score, ones)

    # gold summary loss
    pos_score = rank
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones(pos_score.size()).cuda(score.device)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss += loss_func(pos_score, neg_score, ones)

    return TotalLoss

def edu_ranking_loss(score, margin=0.001):
    # equivalent to initializing TotalLoss to 0
    # here is to avoid that some special samples will not go into the following for loop
    ones = torch.ones(score.size()).cuda(score.device)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)

    # candidate loss
    n = score.size(1)
    for i in range(1, n):
        pos_score = score[:, :-i]
        neg_score = score[:, i:]
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(margin * i)
        TotalLoss += loss_func(pos_score, neg_score, ones)

    return TotalLoss






def precision_query_edus(predict_edu_rank, all_edus_rank, k=10):
    k = min(predict_edu_rank.shape[1], k)
    if predict_edu_rank.shape[1]<=1:
        return torch.as_tensor(1.0, device=predict_edu_rank.device)
    
    s = len(set(predict_edu_rank[0,:k].cpu().numpy()) & set(all_edus_rank[0,:k].cpu().numpy()))/k
    return torch.as_tensor(s, device=predict_edu_rank.device)


def precision_filtering_edus(predict_edu_rank, all_edus_rank, k=10):
    k = min(predict_edu_rank.shape[1], k)
    if predict_edu_rank.shape[1]<=1:
        return torch.as_tensor(1.0, device=predict_edu_rank.device)
    
    s = len(set(predict_edu_rank[0,-k:].cpu().numpy()) & set(all_edus_rank[0,-k:].cpu().numpy()))/k
    return torch.as_tensor(s, device=predict_edu_rank.device)



def compute_mrr_pst_pos(predicted_rank, scores, pos=1):
    """
    @input: 
        predicted_rank: [1, n_docs]
        labels: [1, n_docs]
        pos: position to be validate, pos=0: compute_mrr_1st_pos
            pos=1: the second position
    @example:
        true rank: [[3, 0, 2, 1, 4]],
        predicted_rank: [[3, 1, 0, 2, 4]]
        r
    """

    true_rank = torch.argsort(scores, descending=True)
    if true_rank.shape[1] <= 1:
        return torch.as_tensor(1.0, device=predicted_rank.device)
    if true_rank.shape[1] <= pos:
        return torch.as_tensor(1.0, device=predicted_rank.device)
    
    if true_rank[0,pos] not in predicted_rank:
        return torch.as_tensor(1.0, device=predicted_rank.device)

    rr = 1/(torch.abs(torch.where(predicted_rank==true_rank[0,pos])[1][0]-pos)+1)
    return rr


def compute_precision_at_k(predicted_rank, scores, k=2):
    """
    @input: 
        predicted_rank: [1, n_docs]
        labels: 
    """
    top_k = predicted_rank[:, :k]
    top_k_scores = scores.gather(1, top_k)
    
    precision_at_k = top_k_scores.sum(dim=1) / min(k, predicted_rank.shape[1])
    return precision_at_k.mean()
    
def compute_ndcg(predicted_rank, labels, k=2):
    sorted_labels = labels.gather(1, predicted_rank)

    # Compute DCG
    gains = (2 ** sorted_labels - 1)
    discounts = torch.log2(torch.arange(2, sorted_labels.size(1) + 2, device=labels.device).float())
    dcg = (gains[:, :k] / discounts[:k]).sum(dim=1)

    # Compute Ideal DCG
    ideal_labels, _ = labels.sort(dim=1, descending=True)
    ideal_gains = (2 ** ideal_labels - 1)
    ideal_dcg = (ideal_gains[:, :k] / discounts[:k]).sum(dim=1)

    # Compute NDCG
    ndcg = dcg / (ideal_dcg + 1e-8)
    ndcg = ndcg.mean()
    return ndcg

def ndcg_of_1st_2rd_doc(predicted_rank, scores):
    labels = torch.argsort(scores, dim=1, descending=True)
    docs_num = predicted_rank.shape[1]

    positions = torch.zeros_like(labels) # label j's position in predictions
    for i in range(predicted_rank.shape[0]):
        for j in range(predicted_rank[i].shape[0]):
            positions[i][j] = (labels[i][j]==predicted_rank[i]).nonzero()[0]
    
    relavance_1st = torch.tensor(np.linspace(1,0,docs_num)) # relavance score for 1st doc
    relavance_2st = torch.tensor(np.linspace(1,0,docs_num)) # relavance score for 2rd doc, position in 2rd is highist
    temp = relavance_2st[1]
    relavance_2st[1] = relavance_2st[0]
    relavance_2st[0] = temp

    positions_1_2_st = positions[:,[0,1]]

    relavances = torch.stack((relavance_1st[positions[:,0]], relavance_2st[positions[:,1]])).T
    # DCG
    discounts = torch.log2(1+torch.arange(1, positions.shape[1]+1))[positions_1_2_st]
    dcg = (relavances/discounts).sum(dim=1)

    # IDCG
    idcg = (1/np.log2(2)) + (1/np.log2(3))

    ndcg = (dcg/idcg).mean()

    return ndcg









class RankModel(pl.LightningModule):
    def __init__(self, args):
        super(RankModel, self).__init__()
        self.args = args
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.primer_path)

        self.model = LEDForConditionalGeneration.from_pretrained(args.primer_path)
        self.config = LEDConfig.from_pretrained(args.primer_path)
        self.d_model = self.config.d_model
        if self.args.model_encoder == "encoder_only":
            self.encoder = self.model.get_encoder()
        else:
            self.encoder = self.model.led


        self.pad_token_id = self.tokenizer.pad_token_id

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.edu_transform = nn.Sequential(
                                FeedForward(self.d_model, hidden_dim=self.d_model//2), 
                                GELUActivation(), 
                                nn.LayerNorm(self.d_model)
                            )
        self.doc_transform = nn.Sequential(
                                FeedForward(self.d_model, hidden_dim=self.d_model//2), 
                                GELUActivation(), 
                                nn.LayerNorm(self.d_model)
                            )
        self.doc_pooling = GatedPooling(self.d_model)
        self.cross_edu_to_doc = MultiheadAttentionLayer(self.d_model, self.d_model, self.d_model, 4)
        self.edu_classifier = MLPClassifier(self.d_model)
        
        self.use_ddp = args.strategy == "ddp"
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.loop_n_chunk = 4 # number of chunk in loop

    def forward(self, input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id):
        """
        edus: list of chunks, each chunk contain edu spans
        use examples: edus[0][chunk_id], info[0][chunk_id]
        doc_edu_scores: list of length batch size, each element is doc_edu_scores
        """
        input_ids = input_ids.unsqueeze(0) # [bs, #chunks, chunk_size] # chunk_size: 512
        bsz, n_chunk = input_ids.shape[0], input_ids.shape[1]
        # n_docs = np.max(info.cpu().numpy())+1
        all_doc_ids = np.unique(info.cpu().numpy())
        n_docs = len(all_doc_ids)
        ## 1. Each chunk
        input = input_ids.view(bsz*n_chunk, -1) # [bs*#chunks, chunk_size]
        attn = (input != self.pad_token_id).long()
        doc_edu_scores = doc_edu_scores[0]
        doc_edu_num_list = [len(doc_edu_scores[doc_idx]) for doc_idx in range(n_docs)]
        n_all_edus = np.sum(doc_edu_num_list)
        

        ## 2. get token embeddings
        edu_span_embeddings_doc_dict = defaultdict(list) # store edy embedding of each doc
        loop_num = int(np.ceil(bsz*n_chunk/self.loop_n_chunk))
        current_chunk_position = 0
        with torch.no_grad():
            for chunk_id in range(n_chunk):
                loop_input = input[chunk_id].reshape(1, -1)
                loop_attn = (loop_input != self.pad_token_id).long()
                loop_encoder_outputs = self.encoder(
                    input_ids=loop_input, 
                    attention_mask=loop_attn,
                )
                chunk_token_embeddings = loop_encoder_outputs.last_hidden_state # [1, chunk_size, dim]
                chunk_token_embeddings = chunk_token_embeddings.to('cpu')
                chunk_num, chunk_size, d_model = chunk_token_embeddings.shape

                edu = edus[0][chunk_id]
                doc_id = info[0][chunk_id].cpu().item() # in doc id
                chunk_embedding = chunk_token_embeddings.squeeze(0) # [chunk_size, dim]
                # chunk_edu_span_embeddings = self.span_extractor(chunk_embedding.unsqueeze(0), edu.unsqueeze(0)).squeeze(0) # [#spans, dim]
                edu_embedding_list = get_spans_embeddings_with_index_select(chunk_embedding, edu) # list of [dim], length is the edu num in the chunk
                edu_span_embeddings_doc_dict[doc_id].extend(edu_embedding_list) # list of [dim], length is the edu num in the document
        
        ## 3. pooling embedding to document embedding
        # [#docs, dim]
        doc_embeddings_list = []
        all_edu_embeddings_list = []
        # edu_means_list = []
        for doc_idx in range(n_docs):
            doc_id = all_doc_ids[doc_idx]
            doc_edu_embeddings = torch.stack(edu_span_embeddings_doc_dict[doc_id]).to(input_ids.device) # [n_edus, D]
            doc_edu_embeddings_transform = self.edu_transform(doc_edu_embeddings) # [n_edus, D]
            assert len(doc_edu_scores[doc_idx])==doc_edu_embeddings_transform.shape[0], \
                f"sample: {sample_id}, doc_id: {doc_id}, doc_edu_score {len(doc_edu_scores[doc_idx])} is not aligned with doc_edu_embeddings {doc_edu_embeddings_transform.shape[0]}"

            all_edu_embeddings_list.append(doc_edu_embeddings_transform)

            doc_embedding, doc_pooling_edu_weight = self.doc_pooling(doc_edu_embeddings.unsqueeze(0)) # [1, D]
            # doc_embedding = doc_edu_embeddings.mean(0).unsqueeze(0) # [1, D]
            doc_embeddings_list.append(doc_embedding) # [1, D]
            
        doc_embeddings = torch.concatenate(doc_embeddings_list, dim=0) # [#docs, dim]
        doc_embeddings = self.doc_transform(doc_embeddings) # [#docs, dim]
        
        ### cross edus to docs
        # all_doc_edu_logits = []
        all_edu_embeddings_cross_docs_list = []
        for doc_idx in range(n_docs): # for each doc, cross attention edu to all doc embedding
            doc_edu_embeddings = all_edu_embeddings_list[doc_idx].unsqueeze(0) # [1, n_edus, D]
            doc_edu_embeddings_cross_docs, doc_edu_attn_cross_docs = self.cross_edu_to_doc(doc_edu_embeddings, doc_embeddings.unsqueeze(0), doc_embeddings.unsqueeze(0)) # [1, n_edus, D]  # if with cross attention
            # doc_edu_embeddings_cross_docs = doc_edu_embeddings # if no cross attention
            doc_edu_embeddings_cross_docs = doc_edu_embeddings_cross_docs.squeeze(0)
            all_edu_embeddings_cross_docs_list.append(doc_edu_embeddings_cross_docs)
        
        all_edu_embeddings = torch.concatenate(all_edu_embeddings_cross_docs_list, dim=0) # [n_edus, D]
        edu_logits = self.edu_classifier(all_edu_embeddings) # [n_edus, 1]
        assert n_all_edus==edu_logits.shape[0], "edu_logits shape is not aligned with number of edus"
        
        ### E-step, select out k important edus
        values, indices = edu_logits.squeeze().topk(min(10, n_all_edus))

        if n_all_edus==0 or indices.dim()==0 or indices.shape[0]==0:
            return None, None

        tuple_doc_id_edu_id_list = search_doc_edu_id(doc_edu_num_list, indices.cpu().numpy())
        

        edu_querys = []
        for doc_edu_index in tuple_doc_id_edu_id_list:
            doc_id, doc_edu_id = doc_edu_index
            edu_querys.append(all_edu_embeddings_list[doc_id][doc_edu_id])
        edu_querys = torch.stack(edu_querys)        
        
        ## 5. calculate similarity score
        _scores = torch.matmul(doc_embeddings, edu_querys.T) # [#docs, #top_k]


        if torch.isnan(_scores).any():
            print(f"sample: {sample_id} NaN loss detected, doc_embeddings: {torch.isnan(doc_embeddings).any()}, edu_querys: {torch.isnan(edu_querys).any()}")

        return _scores, edu_logits
    
    def dot_attention(self, q, k, v, v_mask=None, dropout=None):
        attention_weights = torch.matmul(q, k.transpose(-1, -2))
        if v_mask is not None:
            extended_v_mask = (1.0 - v_mask.unsqueeze(1)) * -100000.0
            attention_weights += extended_v_mask
        attention_weights = F.softmax(attention_weights, -1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        return output
    
    def make_doc_mean_pool(self, input_ids, info, tokenizer, n_chunk, token_embeddings):
        """
        example: 
        info: tensor([[0, 0, 0, 1, 1, 1, 1, 1, 2]])
        split: array([3, 5, 1]), 
        splits: [0, 3, 8, 9]
        aa: for each chunk, all positions not pad
        bb: list of chunks, each chunk: all token embeddings filtering pad
        """
        split = nn.functional.one_hot(info).sum(1).squeeze(0).detach().cpu().numpy()
        splits = [0] + np.cumsum(split).tolist() # Find the boundary of each document
        aa = [torch.where(input_ids[0][i] != tokenizer.pad_token_id)[0] for i in range(n_chunk)] # Find each chunk's length without padding
        bb = [token_embeddings[i,aa[i]] for i in range(n_chunk)] # Find each chunk's token embeddings without padding

        # Find the boundary of each document and calculate average pooling for each document 
        _mean_pool = [] 
        for i in range(1, len(splits)): # for each doc
            _doc = []
            for c in bb[splits[i-1]:splits[i]]: # for each chunk
                mean_pool = c.mean(0) # c: [# no_pad tokens, dim]
                # print(f"{i}: {splits[i-1]:splits[i]} mean_pool: {mean_pool.shape}")
                _doc.append(mean_pool.unsqueeze(0))
            _doc_mean = torch.cat(_doc)
            
            _mean_pool.append(_doc_mean.mean(0))
        
        return torch.stack(_mean_pool)

    def make_doc_embeddings(self, input_ids, info, tokenizer, n_chunk, token_embeddings):
        """
        example: 
        info: tensor([[0, 0, 0, 1, 1, 1, 1, 1, 2]])
        split: array([3, 5, 1]), 
        splits: [0, 3, 8, 9]
        aa: for each chunk, all positions not pad
        bb: list of chunks, each chunk: all token embeddings filtering pad
        returns [doc, max_tokens of doc, dim]
        """
        split = nn.functional.one_hot(info).sum(1).squeeze(0).detach().cpu().numpy()
        splits = [0] + np.cumsum(split).tolist() # Find the boundary of each document
        # input_ids[0][i]: batch 0, chunk i
        aa = [torch.where(input_ids[0][i] != tokenizer.pad_token_id)[0] for i in range(n_chunk)] # Find each chunk's length without padding
        bb = [token_embeddings[i,aa[i]] for i in range(n_chunk)] # Find each chunk's token embeddings without padding

        _mean_pool = []
        for i in range(1, len(splits)):
            _doc = []
            for c_id, c in enumerate(bb[splits[i-1]:splits[i]]):
                # mean_pool = c.mean(0)
                _doc.append(c) # c: [# no_pad tokens, dim]
            _doc_mean = torch.cat(_doc) # [# tokens, dim]
            _mean_pool.append(_doc_mean) # list of [# doc tokens, dim]

        doc_embeddings = torch.nn.utils.rnn.pad_sequence(_mean_pool, batch_first=True, padding_value=0) # [#docs, max tokens of doc, dim]
        return doc_embeddings

    def make_doc_input(self, input_ids, info, tokenizer):
        split = nn.functional.one_hot(info).sum(1).squeeze(0).detach().cpu().numpy().tolist()
        split_docs = [] 
        
        for chunks in torch.split_with_sizes(input_ids, split, 1):
            ss = [tokenizer.cls_token_id]
            
            for token in chunks.squeeze(0).flatten(0):
                if token in tokenizer.all_special_ids:
                    continue
                ss.append(token)
            ss.append(tokenizer.sep_token_id)
            split_docs.append(torch.tensor(ss))
        docus = torch.nn.utils.rnn.pad_sequence(split_docs, batch_first=True, padding_value=tokenizer.pad_token_id)
        return docus.to(input_ids.device)
    
    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                list(self.edu_transform.parameters()) + list(self.doc_transform.parameters()) + 
                list(self.doc_pooling.parameters()) + list(self.cross_edu_to_doc.parameters()) + 
                list(self.edu_classifier.parameters()), 
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            # optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.args.lr)
            optimizer = torch.optim.Adam(
                list(self.edu_transform.parameters()) + list(self.doc_transform.parameters()) + 
                list(self.doc_pooling.parameters()) + list(self.cross_edu_to_doc.parameters()) + 
                list(self.edu_classifier.parameters()), 
                lr=self.args.lr)
            scheduler = get_constant_schedule(optimizer)
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(self, input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id):
        """
        _scores: doc_prediction scores [batch_size, #docs], batch_size=1
                when multi_query_loss version, _socres: [#docs, #query]
        _edu_scores: edu rank prediction scores [#edus, 1]
        """
        _scores, edu_logits = self.forward(input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id)
        
        if _scores==None or edu_logits==None:
            return None,None,None

        edu_logits = edu_logits.reshape(1, -1) # [1, #edus]
        edu_scores = edu_logits
        all_doc_edu_scores_flatten = []
        for doc_edus in doc_edu_scores[0]:
            all_doc_edu_scores_flatten.extend(doc_edus)
        all_doc_edu_scores_flatten = torch.tensor(all_doc_edu_scores_flatten).to(input_ids.device)
        all_edus_rank = torch.argsort(all_doc_edu_scores_flatten, descending=True).unsqueeze(0) # [1, #edus]
        assert edu_scores.shape[0]==all_edus_rank.shape[0], f"edu_logits shape {edu_scores.shape[0]} is not aligned with number of edus {all_edus_rank.shape[0]}"

        edu_loss = edu_bpr_loss(edu_scores, all_edus_rank)

        if self.args.loss_type == "rll":
            rank_loss = multi_query_listmle_loss(_scores, rank)
        elif self.args.loss_type == "margin":
            rank_loss = ranking_loss(_scores, scores)
        elif self.args.loss_type == "bpr":
            rank_loss = multi_query_bpr_loss(_scores, rank)
        
        if rank_loss != 0:
            self.log("doc_rank_loss", rank_loss, prog_bar=True)
        if edu_loss != 0:
            self.log("edu_rank_loss", edu_loss, prog_bar=True)

        loss = rank_loss + edu_loss
        
        return _scores, edu_scores, loss
    
    def training_step(self, batch, batch_idx):
        """
        input_ids, 
        output_ids:target_token_ids, 
        edus: edu_to_subtokens_mappings, 
        rank: true ranks
        scores: true similarity between summary and docs, [1, n_docs]
        info: chunk_document_indices
        doc_edu_scores: edu's importance score
        """
        input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id = batch
        _, _edu_scores, loss = self.shared_step(input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id)
        # Check if loss is zero
        if loss == None or loss == 0.0:
            # Log zero loss
            self.log("train_loss", loss, on_step=True)
            return None  # Skip backward pass and optimizer step
        else:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            device = next(self.parameters()).device
            tensorboard_logs = {
                "train_loss": loss,
                "lr": lr,
                "input_size": input_ids.numel(),
                "output_size": output_ids.numel(),
                "mem": torch.cuda.memory_allocated(device) / 1024 ** 3
                if torch.cuda.is_available()
                else 0,
            }
            self.log_dict(tensorboard_logs, on_step=True)
            self.log("print_loss", loss, prog_bar=True)
            self.log("chunk_num", input_ids.shape[0], prog_bar=True)
            return loss

    
    def validation_step(self, batch, batch_idx):
        for p in self.encoder.parameters():
            p.requires_grad = False
        if self.args.mode == 'pretrain':
            input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id = batch
        else:
            input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id, tgt = batch
        _scores, _edu_scores, loss = self.shared_step(input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id)
        
        all_doc_edu_scores_flatten = []
        for doc_edus in doc_edu_scores[0]:
            all_doc_edu_scores_flatten.extend(doc_edus)
        all_doc_edu_scores_flatten = torch.tensor(all_doc_edu_scores_flatten).to(input_ids.device)
        all_edus_rank = torch.argsort(all_doc_edu_scores_flatten, descending=True).unsqueeze(0) # [1, #edus]
        

        result = {"vloss": loss}
        if self.args.compute_accuracy:
            # edu accs
            predict_edu_rank = torch.argsort(_edu_scores, descending=True)
            # doc accs
            predicted_rank_multi_query = torch.argsort(_scores, dim=0, descending=True) # [n_docs, n_querys]
            predicted_rank = get_rank_from_rank_multi(predicted_rank_multi_query)
            predicted_rank = torch.tensor(predicted_rank).unsqueeze(0).to(input_ids.device) # [1, n_docs]
            ndcg = compute_ndcg(predicted_rank, scores, k=2)
            ndcg_3 = compute_ndcg(predicted_rank, scores, k=3)
            mrr_1st = compute_mrr_pst_pos(predicted_rank, scores, pos=0)
            mrr_2st = compute_mrr_pst_pos(predicted_rank, scores, pos=1)
            result["ndcg"] = ndcg
            result["ndcg_3"] = ndcg_3
            result["mrr_1st"] = mrr_1st
            result["mrr_2st"] = mrr_2st
            self.log("ndcg", ndcg, sync_dist=True if self.use_ddp else False, prog_bar=True, batch_size=input_ids.shape[1])
            self.log("ndcg_3", ndcg_3, sync_dist=True if self.use_ddp else False, prog_bar=False, batch_size=input_ids.shape[1])
            self.log("mrr_1st", mrr_1st, sync_dist=True if self.use_ddp else False, prog_bar=False, batch_size=input_ids.shape[1])
            self.log("mrr_2st", mrr_2st, sync_dist=True if self.use_ddp else False, prog_bar=False, batch_size=input_ids.shape[1])
            # result["rouge_result"] = self.compute_rouge_batch(reorganized_inputs, output_ids, tgt)
        self.validation_step_outputs.append(result)
        return result
       
    def on_validation_epoch_end(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        outputs = self.validation_step_outputs
        
        vloss = []
        for x in outputs:
            if not isinstance(x["vloss"], torch.Tensor):
                vloss.append(torch.tensor(x["vloss"], device = self.model.device))
            else:
                vloss.append(x["vloss"])
        vloss = torch.stack(vloss).mean()
        self.log("vloss", vloss, sync_dist=True if self.use_ddp else False)
        if self.args.compute_accuracy:
            ndcg = torch.stack([x["ndcg"] for x in outputs]).mean()
            ndcg_3 = torch.stack([x["ndcg_3"] for x in outputs]).mean()
            mrr_1st = torch.stack([x["mrr_1st"] for x in outputs]).mean()
            mrr_2st = torch.stack([x["mrr_2st"] for x in outputs]).mean()
            logs = {"vloss": vloss, 
                    "ndcg": ndcg, "NDCG@3": ndcg_3, 
                    "mrr_1st": mrr_1st, "mrr_2st": mrr_2st}
            print(f"Validation loss: {vloss}, NDCG@2: {ndcg}, NDCG@3: {ndcg_3}, mrr_1st: {mrr_1st}, mrr_2st: {mrr_2st}")
            self.log_dict(logs, sync_dist=self.use_ddp)
            self.validation_step_outputs.clear()
            return {"vloss": vloss, "log": logs, "progress_bar": logs}
        else:
            logs = {"vloss": vloss}
            self.log_dict(logs, sync_dist=self.use_ddp)
            self.validation_step_outputs.clear()
            return {"vloss": vloss, "log": logs, "progress_bar": logs}
        
    def test_step(self, batch, batch_idx):
        for p in self.encoder.parameters():
            p.requires_grad = False
        if self.args.mode == 'pretrain':
            input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id = batch
        else:
            input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id, tgt = batch
        _scores, _edu_scores, loss = self.shared_step(input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id)

        all_doc_edu_scores_flatten = []
        for doc_edus in doc_edu_scores[0]:
            all_doc_edu_scores_flatten.extend(doc_edus)
        all_doc_edu_scores_flatten = torch.tensor(all_doc_edu_scores_flatten).to(input_ids.device)
        all_edus_rank = torch.argsort(all_doc_edu_scores_flatten, descending=True).unsqueeze(0) # [1, #edus]

        # print(f"_edu_scores: {_edu_scores.shape}, doc_edu_scores: {all_edus_rank.shape}")
        result = {"tloss": loss}
        if self.args.compute_accuracy:
            # edu accs
            predict_edu_rank = torch.argsort(_edu_scores, descending=True)
            # doc accs
            predicted_rank_multi_query = torch.argsort(_scores, dim=0, descending=True)
            predicted_rank = get_rank_from_rank_multi(predicted_rank_multi_query)
            predicted_rank = torch.tensor(predicted_rank).unsqueeze(0).to(input_ids.device)
            ndcg = compute_ndcg(predicted_rank, scores, k=2)
            ndcg_3 = compute_ndcg(predicted_rank, scores, k=3)
            mrr_1st = compute_mrr_pst_pos(predicted_rank, scores, pos=0)
            mrr_2st = compute_mrr_pst_pos(predicted_rank, scores, pos=1)
            result["ndcg"] = ndcg
            result["ndcg_3"] = ndcg_3
            result["mrr_1st"] = mrr_1st
            result["mrr_2st"] = mrr_2st
            self.log("ndcg", ndcg, sync_dist=True if self.use_ddp else False, prog_bar=True, batch_size=input_ids.shape[1])
            self.log("ndcg_3", ndcg_3, sync_dist=True if self.use_ddp else False, prog_bar=True, batch_size=input_ids.shape[1])
            self.log("mrr_1st", mrr_1st, sync_dist=True if self.use_ddp else False, prog_bar=True, batch_size=input_ids.shape[1])
            self.log("mrr_2st", mrr_2st, sync_dist=True if self.use_ddp else False, prog_bar=True, batch_size=input_ids.shape[1])
            # result["rouge_result"] = self.compute_rouge_batch(reorganized_inputs, output_ids, tgt)
        self.test_step_outputs.append(result)
        return result
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        tloss = torch.stack([torch.tensor(x["tloss"], device = self.model.device) if not isinstance(x["tloss"], torch.Tensor) else x["tloss"] for x in outputs]).mean()
        # tloss = torch.stack([x["tloss"] for x in outputs]).mean()
        self.log("tloss", tloss, sync_dist=True if self.use_ddp else False)
        if self.args.compute_accuracy:
            # doc accs
            ndcg = torch.stack([x["ndcg"] for x in outputs]).mean()
            ndcg_3 = torch.stack([x["ndcg_3"] for x in outputs]).mean()
            mrr_1st = torch.stack([x["mrr_1st"] for x in outputs]).mean()
            mrr_2st = torch.stack([x["mrr_2st"] for x in outputs]).mean()
            logs = {"tloss": tloss, 
                    "ndcg": ndcg, "NDCG@3": ndcg_3, 
                    "mrr_1st": mrr_1st, "mrr_2st": mrr_2st}
            print(f"Validation loss: {tloss}, NDCG@2: {ndcg}, NDCG@3: {ndcg_3}, mrr_1st: {mrr_1st}, mrr_2st: {mrr_2st}")

            self.log_dict(logs, sync_dist=self.use_ddp)
            self.test_step_outputs.clear()
            return {"tloss": tloss, "log": logs, "progress_bar": logs}
        else:
            logs = {"tloss": tloss}
            self.log_dict(logs, sync_dist=self.use_ddp)
            self.test_step_outputs.clear()
            return {"tloss": tloss, "log": logs, "progress_bar": logs}
