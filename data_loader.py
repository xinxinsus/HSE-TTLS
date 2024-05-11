import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def build_dataset(configs, dataset, mode='train'):
    dataset = MyDataSet(dataset)
    if mode == 'train':
        data_loader = DataLoader(dataset=dataset, batch_size=configs.batch_size, shuffle=True,
                             collate_fn=bert_batch_preprocessing)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                 collate_fn=bert_batch_preprocessing)
    return data_loader


def bert_batch_preprocessing(batch):
    docid_list, clause_list, doc_len_list, clause_len_list, pairs, \
    feq, feq_len, feq_an, feq_mask, feq_seg, fcq, fcq_len, fcq_an, fcq_mask, fcq_seg, fc_num = zip(*batch)

    feq_an, fe_an_mask = get_answer_pad_mask(feq_an)
    fcq_an, fc_an_mask = get_answer_pad_mask(fcq_an)


    feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj \
        = convert_batch(feq, feq_mask, feq_seg, feq_len, clause_len_list, doc_len_list)
    fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj \
        = convert_batch(fcq, fcq_mask, fcq_seg, fcq_len, clause_len_list, doc_len_list)


    return docid_list, clause_list, pairs, \
           feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
           fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask


class MyDataSet(Dataset):
    def __init__(self, pre_data):
        self.docid_list = pre_data.docid_list
        self.clause_list = pre_data.clause_list
        self.doc_len_list = pre_data.doc_len_list
        self.clause_len_list = pre_data.clause_len_list
        self.pairs = pre_data.pairs

        self._f_emo_query = pre_data._f_emo_query  # [1, max_for_emo_len]
        self._f_emo_query_len = pre_data._f_emo_query_len
        self._f_emo_query_answer = pre_data._f_emo_query_answer
        self._f_emo_query_mask = pre_data._f_emo_query_mask
        self._f_emo_query_seg = pre_data._f_emo_query_seg
        self._f_cau_query = pre_data._f_cau_query
        self._f_cau_query_len = pre_data._f_cau_query_len
        self._f_cau_query_answer = pre_data._f_cau_query_answer
        self._f_cau_query_mask = pre_data._f_cau_query_mask
        self._f_cau_query_seg = pre_data._f_cau_query_seg
        self._forward_c_num = pre_data._forward_c_num

    def __len__(self):
        return len(self.doc_len_list)

    def __getitem__(self, i):
        docid_list, clause_list, doc_len_list, clause_len_list, pairs, \
        feq, feq_len, feq_an, feq_mask, feq_seg, fcq, fcq_len, fcq_an, fcq_mask, fcq_seg, fc_num = \
            self.docid_list[i], self.clause_list[i], self.doc_len_list[i], self.clause_len_list[i], self.pairs[i], \
            self._f_emo_query[i], self._f_emo_query_len[i], self._f_emo_query_answer[i], self._f_emo_query_mask[i], self._f_emo_query_seg[i], \
            self._f_cau_query[i], self._f_cau_query_len[i], self._f_cau_query_answer[i], self._f_cau_query_mask[i], self._f_cau_query_seg[i],self._forward_c_num[i]

        return docid_list, clause_list, doc_len_list, clause_len_list, pairs, \
        feq, feq_len, feq_an, feq_mask, feq_seg, fcq, fcq_len, fcq_an, fcq_mask, fcq_seg, fc_num




def get_answer_pad_mask(answer):
    new_answer = []
    for batch_answer in answer:
        for qa_answer in batch_answer:
            new_answer.append(torch.tensor(qa_answer))
    answer = pad_sequence(new_answer, padding_value=-1).transpose(0, 1)
    mask = torch.where(answer != -1, torch.ones_like(answer), torch.zeros_like(answer))
    assert mask.shape == answer.shape
    return answer, mask


def convert_batch(query, query_mask, query_seg, query_len, seq_len, doc_len):
    query_list, query_mask_list, query_seg_list = [], [], []
    new_query_len, new_seq_len, new_doc_len = [], [], []
    for i in range(len(query_len)):
        for j in range(len(query_len[i])):
            query_list.append(query[i][j])
            query_mask_list.append(query_mask[i][j])
            query_seg_list.append(query_seg[i][j])
            new_seq_len.append(seq_len[i])
            new_doc_len.append(doc_len[i])
            new_query_len.append(query_len[i][j])

    query = torch.LongTensor(query_list)
    query_mask = torch.LongTensor(query_mask_list)
    query_seg = torch.LongTensor(query_seg_list)
    adj = pad_matrices(new_doc_len)
    return query, query_mask, query_seg, new_query_len, new_seq_len, new_doc_len, adj



def pad_matrices(doc_len_b):
    try:
        N = max(doc_len_b)
    except:
        N=0
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


