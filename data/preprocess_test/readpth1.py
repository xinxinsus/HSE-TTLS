import  torch
from torch.utils.data import Dataset, DataLoader
class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self.docid_list = pre_data['_docid_list']
        self.clause_list = pre_data['_clause_list']
        self.doc_len_list = pre_data['_doc_len_list']
        self.clause_len_list = pre_data['_clause_len_list']
        self.pairs = pre_data['_pairs']

        self._f_emo_query = pre_data['_f_emo_query']  # [1, max_for_emo_len]
        self._f_cau_query = pre_data['_f_cau_query']  # [max_for_num, max_for_cau_len]
        self._f_emo_query_len = pre_data['_f_emo_query_len']
        self._f_cau_query_len = pre_data['_f_cau_query_len']
        self._f_emo_query_answer = pre_data['_f_emo_query_answer']
        self._f_cau_query_answer = pre_data['_f_cau_query_answer']
        self._f_emo_query_mask = pre_data['_f_emo_query_mask']  # [1,max_for_emo_len]
        self._f_cau_query_mask = pre_data['_f_cau_query_mask']  # [max_for_num, max_for_cau_len]
        self._f_emo_query_seg = pre_data['_f_emo_query_seg']  # [1,max_for_emo_len]
        self._f_cau_query_seg = pre_data['_f_cau_query_seg']  # [max_for_num, max_for_cau_len]

        self._b_emo_query = pre_data['_b_emo_query']
        self._b_cau_query = pre_data['_b_cau_query']  #
        self._b_emo_query_len = pre_data['_b_emo_query_len']
        self._b_cau_query_len = pre_data['_b_cau_query_len']
        self._b_emo_query_answer = pre_data['_b_emo_query_answer']
        self._b_cau_query_answer = pre_data['_b_cau_query_answer']
        self._b_emo_query_mask = pre_data['_b_emo_query_mask']  #
        self._b_cau_query_mask = pre_data['_b_cau_query_mask']  #
        self._b_emo_query_seg = pre_data['_b_emo_query_seg']  #
        self._b_cau_query_seg = pre_data['_b_cau_query_seg']  #

        self._forward_c_num = pre_data['_forward_c_num']
        self._backward_e_num = pre_data['_backward_e_num']

content=torch.load("fold1.pt")
for k, v in content.items():  # k 参数名 v 对应参数值
    print(k)
    print(v)
print(content.keys())

