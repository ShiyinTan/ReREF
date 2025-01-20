from collections import defaultdict
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from pathlib import Path
import torch
from random import shuffle
import random
import os
from nltk.tokenize import sent_tokenize
import re
import sys
import spacy
import re



def get_docs_euds_lists(docs):
    """
    get doc_clean_list and doc_edus_list from docs string.
    Args:
        docs (string): use ' || ' to split edus, and use ' ||||| ' to split documents.
    Return:
        doc_clean_list, doc_edus_list
    """
    doc_list = docs.split(' ||||| ')
    doc_clean_list = []
    doc_edus_list = []
    for i, doc in enumerate(doc_list):
        doc = doc.lstrip()
        edus = doc.split(' || ')
        doc_edus_list.append(edus)
        clean_doc = " ".join(edus)
        clean_doc = re.sub(r'\s+', ' ', clean_doc) # clean multiple spaces into one
        doc_clean_list.append(clean_doc)
    return doc_clean_list, doc_edus_list







def chunking_and_extract_edu_spans(entry, tokenizer, chunk_size=512, dataset_name="multi_news"):
    """
    example: 
    input: chunk_id, span_id
    doc_id = chunk_document_indices[chunk_id]
    edu_span = edu_to_subtokens_mappings[chunk_id][span_id]
    edu_ids = inputs_ids[chunk_id][edu_span[0]:edu_span[1]]
    @input: 
        docs: documents string split by ' ||||| ', and edu split by ' || '
        tokenizer
    @output:
        input_ids: list of chunks, each chunk contains chunk_input_ids list
        edu_to_subtokens_mappings: list of chunks, each chunk contains the tuple list of edu span
        chunk_document_indices: list of chunks, each chunk contains the doc id 
        final_doc_edu_scores: some edu will be droped, so we only record the contained edu scores
    """
    
    
    sample_id = entry['id']
    docs = entry['document']
    doc_scores = entry["sent_trans_doc_score"]
    doc_edu_scores = entry["sent_trans_doc_edu_score"]

    start_token_id = tokenizer.bos_token_id
    end_token_id = tokenizer.eos_token_id

    doc_clean_list, doc_edus_list = get_docs_euds_lists(docs)
    
    final_doc_edu_scores = []
    chunk_id = 0
    input_ids = defaultdict(lambda: list([start_token_id]))
    edu_to_subtokens_mappings = defaultdict(list)
    chunk_document_indices = []
    for doc_idx, edu_info in enumerate(doc_edus_list): # for each doc, and corresponding edus
        # edu_info = [edu.strip() for edu in edu_info if edu.strip()] # remove empty/space string
        cureent_edu_scores_in_doc = []
        for idx, edu in enumerate(edu_info): # for each edu in doc
            edu_score = doc_edu_scores[doc_idx][idx]
            if edu.strip() == '':
                continue

            if idx != 0:
                edu_ids = tokenizer.encode(" "+edu.strip(), add_special_tokens=False)
            else:
                edu_ids = tokenizer.encode(edu.strip(), add_special_tokens=False)
            edu_len = len(edu_ids)

            cureent_edu_scores_in_doc.append(edu_score)
            if len(input_ids[chunk_id]) + len(edu_ids) +1 >= chunk_size:  # if chunk token number larger than chunk size
                chunk_id += 1
                chunk_document_indices.append(doc_idx)
            
            edu_spans = (len(input_ids[chunk_id]), len(input_ids[chunk_id]) + len(edu_ids))
            edu_to_subtokens_mappings[chunk_id].append(edu_spans)
            input_ids[chunk_id].extend(edu_ids)
        
        if len(cureent_edu_scores_in_doc)==0: # the doc is empty, skip
            continue
        
        input_ids[chunk_id].append(end_token_id) # at end of single document, append a <eos>
        chunk_id += 1
        chunk_document_indices.append(doc_idx)
        final_doc_edu_scores.append(cureent_edu_scores_in_doc)
    
    input_ids_result = [torch.tensor(input_ids[key]) for key in input_ids.keys()]
    edu_to_subtokens_mappings_result = [edu_to_subtokens_mappings[key] for key in edu_to_subtokens_mappings.keys()]
    return input_ids_result, edu_to_subtokens_mappings_result, chunk_document_indices, final_doc_edu_scores




class SummarizationDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        dataset_name,
        join_method,
        tokenizer,
        max_input_len,
        max_output_len,
        chunk_size,
        mask_num=5,
        num_data=-1,
        rand_seed=1,
        is_test=False,
        dataset_type="train",
    ):
        self.hf_dataset = hf_dataset
        self.dataset_name = dataset_name
        self.join_method = join_method
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.chunk_size = chunk_size
        if join_method in  "concat_start_wdoc_global":
            self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        self.mask_id = self.tokenizer.mask_token_id
        self.mask_num = mask_num
        if num_data != -1 and not is_test and num_data < len(hf_dataset):
            random.seed(rand_seed)
            self.hf_dataset = hf_dataset.select(np.random.permutation(len(hf_dataset))[:num_data])
        
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        
        # single doc setting
        if self.dataset_name == "pubmed":
            src = entry["article"]
            tgt = entry["abstract"]
            input_ids = self.tokenizer.encode(
                src, truncation=True, max_length=self.max_input_len
            )
            output_ids = self.tokenizer.encode(
                tgt, truncation=True, max_length=self.max_output_len
            )
        else:  # multi-doc setting
            
            document = entry["document"]
            tgt = entry["summary"]
            sample_id = entry["id"]
            doc_scores = entry["sent_trans_doc_score"]


            mask_num = self.mask_num
            input_ids, edus, info, doc_edu_scores = chunking_and_extract_edu_spans(entry, self.tokenizer, chunk_size=self.chunk_size, dataset_name=self.dataset_name)

            """
            the rank is calculated on whole document,  if we only use top 10 documents, the rank will change.
            For example, the actual rank was [13,5,6,10,7,~~], but if we consider only top 10. 
            we don't need to consider the 13 document which is top 1 in original rank.
            """
            all_doc_ids = np.unique(info) # some doc may be dropped, and be empty
            doc_scores = [doc_scores[doc_id] for doc_id in all_doc_ids]
            doc_rank = np.argsort(doc_scores)[::-1].tolist() # decrease order rank

            labels = [0] * len(np.unique(np.array(info)).tolist())
            labels[doc_rank[0]] = 1

                               
            output_ids = self.tokenizer.encode(
                tgt, truncation=True, max_length=self.tokenizer.model_max_length, padding="max_length"
            )

        

        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        if self.dataset_type == "train":
            
            return input_ids, torch.tensor(output_ids), edus, doc_rank, doc_scores, info, labels, doc_edu_scores, sample_id
        else:
            return input_ids, torch.tensor(output_ids), edus, doc_rank, doc_scores, info, labels, doc_edu_scores, sample_id, tgt


class PretrainDataset(IterableDataset):
    def __init__(
        self,
        inputs_dir,
        dataset_type,
        max_input_len,
        max_output_len,
        use_ddp=False,
        remove_masks=False,
        mask_id=0,
    ):
        super().__init__()
        if isinstance(inputs_dir, list):
            self._input_files = inputs_dir
        else:
            inputs_dir = Path(os.path.join(inputs_dir, dataset_type))
            self._input_files = [path for path in inputs_dir.glob("*.pt")]
        self.shuffle = dataset_type == "train"
        self._input_files = sorted(self._input_files)
        if self.shuffle:
            self._shuffle()
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.start = 0
        self.end = len(self._input_files)
        self.use_ddp = use_ddp
        self.remove_masks = remove_masks
        self.mask_id = mask_id

    def _loaddata(self, idx):
        file = self._input_files[idx]
        cur_data = torch.load(file)
        if self.shuffle:
            shuffle(cur_data)
        return cur_data

    def _shuffle(self):
        # shuffle the list of data files after each epoch
        shuffle(self._input_files)

    def _set_worker(self):
        # The whole dataset covering all the files in self._input_files
        overall_start = 0
        overall_end = len(self._input_files)

        # Get the worker id in the current world
        worker_info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (
            (worker_info.num_workers, worker_info.id)
            if worker_info is not None
            else (1, 0)
        )

        # Get the worker id in the overall worlds
        if self.use_ddp:
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            worker_global_rank = global_rank * num_workers + worker_id
        else:
            worker_global_rank = worker_id
            world_size = 1

        # Get the total number of workers and split tasks accordingly
        worker_world_size = num_workers * world_size
        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_world_size))
        )

        # Set the current task, based on overall worker id and task splitting
        self.start = overall_start + worker_global_rank * per_worker
        self.end = min(self.start + per_worker, overall_end)

    def __iter__(self):
        self._set_worker()
        all_indices = list(range(self.start, self.end))
        if self.shuffle:
            # use inifinite iterators for training
            while True:
                shuffle(all_indices)
                for i in all_indices:
                    print("datafile is ", self._input_files[i])
                    sys.stdout.flush()
                    cur_data = self._loaddata(i)
                    while len(cur_data) != 0:
                        # print('data index is ', len(cur_data))
                        data = cur_data.pop()
                        if self.remove_masks:
                            data["src"] = list(
                                filter(lambda a: a != self.mask_id, data["src"])
                            )
                            # print(data["src"])
                        if len(data["src"]) > self.max_input_len:
                            data["src"] = data["src"][: (self.max_input_len - 1)] + [
                                data["src"][-1]
                            ]  # add </s>
                        if len(data["tgt"]) > self.max_output_len:
                            data["tgt"] = data["tgt"][: (self.max_output_len - 1)] + [
                                data["tgt"][-1]
                            ]  # add </s>
                        yield torch.tensor(data["src"]), torch.tensor(data["tgt"])
        else:
            # use normal iterators for validation
            for i in all_indices:
                print("datafile is ", self._input_files[i])
                sys.stdout.flush()
                cur_data = self._loaddata(i)
                while len(cur_data) != 0:
                    # print('data index is ', len(cur_data))
                    data = cur_data.pop()
                    if self.remove_masks:
                        data["src"] = list(
                            filter(lambda a: a != self.mask_id, data["src"])
                        )
                        # print(data["src"])
                    if len(data["src"]) > self.max_input_len:
                        data["src"] = data["src"][: (self.max_input_len - 1)] + [
                            data["src"][-1]
                        ]  # add </s>
                    if len(data["tgt"]) > self.max_output_len:
                        data["tgt"] = data["tgt"][: (self.max_output_len - 1)] + [
                            data["tgt"][-1]
                        ]  # add </s>
                    yield torch.tensor(data["src"]), torch.tensor(data["tgt"])


class SummarizationIterDataset(IterableDataset):
    def __init__(
        self,
        join_method,
        dataset_name,
        tokenizer,
        inputs_dir,
        dataset_type,
        max_input_len,
        max_output_len,
        use_ddp=False,
        mask_num=0,
    ):
        super().__init__()
        if isinstance(inputs_dir, list):
            self._input_files = inputs_dir
        else:
            inputs_dir = Path(os.path.join(inputs_dir, dataset_type))
            self._input_files = [path for path in inputs_dir.glob("*.pt")]
        self._input_files = sorted(self._input_files)
        self.shuffle = dataset_type == "train"
        if self.shuffle:
            self._shuffle()
        self.join_method = join_method
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.start = 0
        self.end = len(self._input_files)
        self.use_ddp = use_ddp
        if join_method == "concat_start_wdoc_global":
            self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.mask_num = mask_num
        self.dataset_type = dataset_type

    def _loaddata(self, idx):
        file = self._input_files[idx]
        cur_data = torch.load(file)
        if self.shuffle:
            shuffle(cur_data)
        return cur_data

    def _shuffle(self):
        # shuffle the list of data files after each epoch
        shuffle(self._input_files)

    def _set_worker(self):
        # The whole dataset covering all the files in self._input_files
        overall_start = 0
        overall_end = len(self._input_files)

        # Get the worker id in the current world
        worker_info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (
            (worker_info.num_workers, worker_info.id)
            if worker_info is not None
            else (1, 0)
        )

        # Get the worker id in the overall worlds
        if self.use_ddp:
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            worker_global_rank = global_rank * num_workers + worker_id
        else:
            worker_global_rank = worker_id
            world_size = 1

        # Get the total number of workers and split tasks accordingly
        worker_world_size = num_workers * world_size
        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_world_size))
        )

        # Set the current task, based on overall worker id and task splitting
        self.start = overall_start + worker_global_rank * per_worker
        self.end = min(self.start + per_worker, overall_end)

    def __iter__(self):
        self._set_worker()

        for i in range(self.start, self.end):
            print("datafile is ", i)
            cur_data = self._loaddata(i)
            while len(cur_data) != 0:
                # print("data index is ", len(cur_data))
                data = cur_data.pop()
                all_docs = data["text"]
                if self.join_method == "plain_concat":
                    src = "\n".join(all_docs)
                    tgt = data["tgt"]
                    input_ids = self.tokenizer.encode(
                        src, truncation=True, max_length=self.max_input_len
                    )
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_eachdoc":
                    input_ids = []
                    for doc in all_docs:
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=self.max_input_len // len(all_docs),
                            )[1:-1]
                        )
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    tgt = data["tgt"]
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_eachdoc_wsent_global":
                    input_ids = []
                    for doc in all_docs:
                        sents = [
                            " [sent] ".join(sent_tokenize(p)) + " [sent]"
                            for p in doc.split("\n")
                            if p != ""
                        ]
                        doc = "\n".join(sents)
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=self.max_input_len // len(all_docs),
                            )[1:-1]
                        )
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    tgt = data["tgt"]
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_wdoc_global":
                    mask_num = self.mask_num
                    tgt = data["tgt"]

                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    for doc in all_docs:
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=(self.max_input_len - mask_num)
                                // len(all_docs),
                            )[1:-1]
                        )
                        input_ids.append(self.docsep_token_id)
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )

                if self.tokenizer.bos_token_id is None:  # pegasus
                    output_ids = [self.tokenizer.pad_token_id] + output_ids
                    input_ids = input_ids[1:]
                if self.dataset_type == "train":
                    yield torch.tensor(input_ids), torch.tensor(output_ids)
                else:
                    yield torch.tensor(input_ids), torch.tensor(output_ids), tgt


def collate_fn(batch):
    pad_token_id = 1 # LED
    train = True
    if len(batch[0]) == 10:
        train = False
        tgt = [item[9] for item in batch]
        edus = [item[2] for item in batch]
        rank = [item[3] for item in batch]
        scores = [item[4] for item in batch]
        info = [item[5] for item in batch]
        labels = [item[6] for item in batch]
        doc_edu_scores = [item[7] for item in batch]
        sample_id = [item[8] for item in batch]
        batch = [item[:9] for item in batch]
    
    input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id = list(zip(*batch))
    
    input_ids = torch.vstack([
        torch.nn.utils.rnn.pad_sequence(
            input, 
            batch_first=True, padding_value=pad_token_id) 
            for input in input_ids])

    padded_input_ids = []
    for input in input_ids:
        padding_len = (512 - input.shape[-1] % 512) % 512
        padded_input_ids.append(torch.nn.functional.pad(input, (0, padding_len), value=pad_token_id))
    
    input_ids = torch.vstack(padded_input_ids)
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    
    rank = torch.tensor(rank)
    scores = torch.tensor(scores)
    info = torch.tensor(info)
    labels = torch.tensor(labels)
    if train:
        return input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id
    else:
        return input_ids, output_ids, edus, rank, scores, info, labels, doc_edu_scores, sample_id, tgt


def get_dataloader_summ(
    args, hf_datasets, tokenizer, split_name, num_workers, is_train
):
    d = hf_datasets[split_name]

    dataset = SummarizationDataset(
        hf_dataset=d,
        dataset_name=args.dataset_name,
        join_method=args.join_method,
        tokenizer=tokenizer,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        chunk_size=args.chunk_size,
        mask_num=args.mask_num,
        num_data=args.num_train_data,
        rand_seed=args.rand_seed,
        is_test=(split_name == "test"),
        dataset_type=split_name,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        # sampler=sampler,
        collate_fn=collate_fn,
    )


def get_dataloader_pretrain(
    args, inputs_dir, dataset_type, num_workers, use_ddp=False, mask_id=0
):
    dataset = PretrainDataset(
        inputs_dir,
        dataset_type,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        use_ddp=use_ddp,
        remove_masks=args.remove_masks,
        mask_id=mask_id,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def get_dataloader_summiter(
    args, tokenizer, inputs_dir, dataset_type, num_workers, use_ddp=False
):
    dataset = SummarizationIterDataset(
        args.join_method,
        args.dataset_name,
        tokenizer,
        inputs_dir,
        dataset_type,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        use_ddp=use_ddp,
        mask_num=args.mask_num,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )