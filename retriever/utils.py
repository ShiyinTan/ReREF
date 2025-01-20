from collections import Counter
import torch
import numpy as np

def search_doc_edu_id(token_count_list, index_list):
    """
    token_count_list: list of each doc's edu number
    index_list: the index of edu in all_edu_list
    return: a list of tuple (doc_id, in_doc_edu_id), each item present the important edu position.
    """
    doc_id_edu_id_list = []
    for index in index_list:
        doc_id = np.searchsorted(np.cumsum(token_count_list), index+1)
        if doc_id == 0:
            in_doc_edu_id = index
        else:
            in_doc_edu_id = index - np.cumsum(token_count_list)[doc_id-1]
        assert doc_id < len(token_count_list), "doc_id out of index"
        assert in_doc_edu_id >= 0 and in_doc_edu_id < token_count_list[doc_id], "edu_id out of index"
        doc_id_edu_id_list.append((doc_id, in_doc_edu_id))
    return doc_id_edu_id_list

def get_rank_from_rank_multi(rank_multi_query):
    """
    return list of rank results: i.e. [3, 1, 2, 5]
    """
    n_docs, n_query = rank_multi_query.shape # [n_docs, n_query]
    most_common_items = []
    rank_results = []
    for row in rank_multi_query:
        counts = torch.bincount(row, minlength=rank_multi_query.max()+1)
        max_count = counts.max()  # 找到最大次数
        most_common = torch.where(counts == max_count)[0]  # 找到所有出现次数等于最大次数的元素
        most_common_items.append(most_common)
        rank_results.extend(most_common.cpu().numpy().tolist())
    rank_results = list(dict.fromkeys(rank_results))
    if len(rank_results) < n_docs: # if rank result is shorter than n_docs
        final_rank_results = rank_results + np.arange(len(rank_results), n_docs).tolist()
    else:
        final_rank_results = rank_results
    return final_rank_results

def get_spans_embeddings_with_index_select(token_embedding, spans, method='mean', device='cpu'):
    """
    使用 `index_select` 来获取多个span的embedding.
    
    Args:
        token_embedding (torch.Tensor): 形状为 [seq_len, d_model] 的token embedding.
        spans (List[Tuple[int, int]]): 一个包含span的列表，每个span是 (start_idx, end_idx).
        method (str): 对span的处理方法，默认是 'mean' (可以选择 'mean' 或 'max').
        
    Returns:
        List[torch.Tensor]: 每个span的embedding，返回一个列表，每个元素形状为 [d_model].
    """
    span_embeddings = []
    
    for start_idx, end_idx in spans:
        # 获取 span 范围内的 token embeddings
        indices = torch.arange(start_idx, end_idx, device=token_embedding.device)
        
        # 通过 index_select 选择相关的 token embeddings
        span_embedding = torch.index_select(token_embedding, dim=0, index=indices)
        
        # 根据选择的 method 处理 span embedding
        if method == 'mean':
            span_embedding = span_embedding.mean(dim=0)
        elif method == 'max':
            span_embedding, _ = span_embedding.max(dim=0)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'mean', 'max'")
        
        # 将计算好的 span embedding 添加到列表
        span_embeddings.append(span_embedding.to(device))
    
    return span_embeddings



def tokenize(text):
    # Convert to lowercase and split into words
    return text.lower().split()


def common_word_count(str1, str2):
    # Tokenize both strings
    words1 = tokenize(str1)
    words2 = tokenize(str2)
    
    # Count words in both strings
    counter1 = Counter(words1)
    counter2 = Counter(words2)
    
    # Find the common words
    common_words = counter1 & counter2
    
    # Return the number of common words
    return sum(common_words.values())

def remove_duplicates(strings, threshold=7):
    # Create a list to store unique strings
    unique_strings = []
    
    # Iterate through each string in the list
    for i, string in enumerate(strings):
        is_duplicate = False
        
        # Compare with previously added unique strings
        for unique_string in unique_strings:
            if common_word_count(string, unique_string) > threshold:
                is_duplicate = True
                break
        
        # If not a duplicate, add to unique strings
        if not is_duplicate:
            unique_strings.append(string)
    
    return unique_strings

def get_device_of(tensor: torch.Tensor):
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()

def get_range_vector(size, device):
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def batched_span_select(target, spans):
    span_starts, span_ends = spans.split(1, dim=-1)
    span_widths = span_ends - span_starts
    max_batch_span_width = span_widths.max().item() + 1
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    span_mask = (max_span_range_indices <= span_widths).float()
    raw_span_indices = span_ends - max_span_range_indices
    span_mask = span_mask * (raw_span_indices >= 0).float()
    span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()
    span_embeddings = batched_index_select(target, span_indices)
    return span_embeddings, span_mask


def flatten_and_batch_shift_indices(indices, sequence_length):
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ValueError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)
    offset_indices = indices + offsets
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target, indices, flattened_indices=None):
    if flattened_indices is None:
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))
    flattened_target = target.view(-1, target.size(-1))
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def weighted_sum(matrix, attention):
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class TimeDistributed(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs, pass_through=None, **kwargs):
        pass_through = pass_through or []
        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]
        some_input = None
        if inputs:
            some_input = inputs[-1]
        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value
                value = self._reshape_tensor(value)
            reshaped_kwargs[key] = value
        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)
        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")
        new_size = some_input.size()[:2] + reshaped_outputs.size()[1:]
        outputs = reshaped_outputs.contiguous().view(new_size)
        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)



class SelfAttentiveSpanExtractor(torch.nn.Module):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Registered as a `SpanExtractor` with name "self_attentive".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.

    # Returns

    attended_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """
    def __init__(self, input_dim):
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._input_dim

    def forward(self, sequence_tensor, span_indices, span_indices_mask=None):
        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)
        # shape (batch_size, sequence_length, embedding_dim + 1)
        concat_tensor = torch.cat([sequence_tensor, global_attention_logits], -1)
        concat_output, span_mask = batched_span_select(concat_tensor, span_indices)
        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = masked_softmax(span_attention_logits, span_mask)
        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = weighted_sum(span_embeddings, span_attention_weights)
        if span_indices_mask is not None:
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1).float()
        return attended_text_embeddings