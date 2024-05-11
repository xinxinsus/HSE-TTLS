import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

def data_bert_trunk(doc_len, doc_couples, clauses):
    sum_len = 0
    for clause in clauses:
        sum_len += (2 + len(clause))  # [CLS], [SEP]
    if sum_len > 375:  # trunk
        pair = doc_couples[0]
        half_len = doc_len // 2
        if pair[0] <= half_len and pair[1] <= half_len:  # 取文档前半部分
            doc_len = half_len
            clauses = clauses[: half_len]
        else:   # 取文档后半部分
            doc_len = doc_len - half_len
            for i in range(len(doc_couples)):
                doc_couples[i][0] -= half_len
                doc_couples[i][1] -= half_len
            clauses = clauses[half_len: ]

    assert doc_len == len(clauses)
    return doc_len, doc_couples, clauses

class dual_sample(object):
    def __init__(self, doc_id, doc_len, text, clause_list, clause_len_list, pairs, f_query_len_list,
                 forward_query_list,  f_e_query_answer, f_c_query_answer_list):
        self.doc_id = doc_id
        self.doc_len = doc_len
        self.text = text
        self.clause_list = clause_list
        self.clause_len_list = clause_len_list
        self.pairs = pairs

        self.f_query_list = forward_query_list
        self.f_query_len_list = f_query_len_list
        self.f_e_query_answer = f_e_query_answer
        self.f_c_query_answer = f_c_query_answer_list


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
        self._forward_c_num = pre_data['_forward_c_num']



def pre_processing(sample_list, max_len_dict):
    tokenizer = BertTokenizer.from_pretrained('/pretrained_model/bert-base-chinese')
    _docid_list = []
    _clause_list = []
    _doc_len_list = []
    _clause_len_list = []
    _pairs = []

    _f_emo_query = []
    _f_cau_query = []
    _f_emo_query_len = []
    _f_cau_query_len = []
    _f_emo_query_answer = []
    _f_cau_query_answer = []
    _f_emo_query_mask = []
    _f_cau_query_mask = []
    _f_emo_query_seg = []
    _f_cau_query_seg = []
    _forward_c_num = []


    for instance in sample_list:  # 对每一个文档
        _docid_list.append(instance.doc_id)
        _clause_list.append(instance.clause_list)
        _doc_len_list.append(instance.doc_len)
        _clause_len_list.append(instance.clause_len_list)
        _pairs.append(instance.pairs)
        _f_emo_query_len.append(instance.f_query_len_list[0:1])
        _f_cau_query_len.append(instance.f_query_len_list[1:])

        f_query_list = instance.f_query_list
        f_query_seg_list = instance.f_query_seg

        _forward_c_num.append(len(f_query_list) - 1)

        # For EE
        f_single_emotion_query = []
        f_single_emotion_query_mask = []
        f_single_emotion_query_seg = []
        f_emo_pad_num = max_len_dict['max_f_emo_len'] - len(f_query_list[0])
        # query
        f_single_emotion_query.append(tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[0]])
                                  + [0] * f_emo_pad_num)
        # mask
        f_single_emotion_query_mask.append([1] * len(f_query_list[0]) + [0] * f_emo_pad_num)
        # segment
        f_single_emotion_query_seg.append(f_query_seg_list[0] + [1] * f_emo_pad_num)

        _f_emo_query_answer.append(instance.f_e_query_answer)
        _f_emo_query.append(f_single_emotion_query)
        _f_emo_query_seg.append(f_single_emotion_query_seg)
        _f_emo_query_mask.append(f_single_emotion_query_mask)

        # For ECE
        f_single_cause_query = []
        f_single_cause_query_mask = []
        f_single_cause_query_seg = []
        for i in range(1, len(f_query_list)):
            pad_num = max_len_dict['max_f_cau_len'] - len(f_query_list[i])
            # query
            f_single_cause_query.append(tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[i]])
                                      + [0] * pad_num)
            # mask
            f_single_cause_query_mask.append([1] * len(f_query_list[i]) + [0] * pad_num)
            # segment
            f_single_cause_query_seg.append(f_query_seg_list[i] + [1] * pad_num)
            assert len(f_single_cause_query[-1]) == len(f_single_cause_query_seg[-1]) == len(f_single_cause_query_mask[-1])
        # PAD: max_f_num
        _f_cau_query.append(f_single_cause_query)
        _f_cau_query[-1].extend([[0] * max_len_dict['max_f_cau_len']] * (max_len_dict['max_f_c_num'] - _forward_c_num[-1]))
        _f_cau_query_mask.append(f_single_cause_query_mask)
        _f_cau_query_mask[-1].extend([[0] * max_len_dict['max_f_cau_len']] * (max_len_dict['max_f_c_num'] - _forward_c_num[-1]))
        _f_cau_query_seg.append(f_single_cause_query_seg)
        _f_cau_query_seg[-1].extend([[0] * max_len_dict['max_f_cau_len']] * (max_len_dict['max_f_c_num'] - _forward_c_num[-1]))

        _f_cau_query_answer.append(instance.f_c_query_answer)



    result =  {'_docid_list': _docid_list, '_clause_list': _clause_list, '_doc_len_list': _doc_len_list,
               '_clause_len_list': _clause_len_list, '_pairs': _pairs,
               '_f_emo_query': _f_emo_query, '_f_cau_query': _f_cau_query,
               '_f_emo_query_len': _f_emo_query_len, '_f_cau_query_len': _f_cau_query_len,
               '_f_emo_query_answer': _f_emo_query_answer, '_f_cau_query_answer': _f_cau_query_answer,
               "_f_emo_query_mask": _f_emo_query_mask, "_f_cau_query_mask": _f_cau_query_mask,
               "_f_emo_query_seg": _f_emo_query_seg, "_f_cau_query_seg": _f_cau_query_seg,
               "_forward_c_num": _forward_c_num}
    return OriginalDataset(result)


def tokenized_data(data):
    max_f_emo_query_len = 0
    max_f_cau_query_len = 0
    max_f_c_querys_num = 0
    max_doc_len = 0
    tokenized_sample_list = []
    for sample in data:  # 对每一个文档
        doc_len = sample.doc_len
        max_doc_len = max(max_doc_len, doc_len)
        f_querys, f_answers, f_querys_seg= [], [], []
        max_f_c_querys_num = max(max_f_c_querys_num, len(sample.f_query_list) - 1)

        for i in range(len(sample.f_query_list)):
            temp_query = sample.f_query_list[i]
            # EE
            if len(temp_query)==0:
                temp_text = sample.text
                temp_text = ' '.join(temp_text).split(' ')
                temp_qa = ['[CLS]'] + temp_text
                temp_qa_seg = [1] * (1+len(temp_text))
            else:
            # ECE
                temp_query =' '.join(temp_query[0]).split(' ')
                temp_text = sample.text
                # text也做同样的操作
                temp_text = ' '.join(temp_text).split(' ')
                temp_qa = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
                temp_qa_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)


            assert len(temp_qa) == len(temp_qa_seg)

            if i == 0:
                max_f_emo_query_len = max(max_f_emo_query_len, len(temp_qa))
            else:
                max_f_cau_query_len = max(max_f_cau_query_len, len(temp_qa))
            f_querys.append(temp_qa)
            f_querys_seg.append(temp_qa_seg)


        sample.f_query_list = f_querys
        sample.f_query_seg = f_querys_seg
        tokenized_sample_list.append(sample)


    return tokenized_sample_list, {'max_f_emo_len': max_f_emo_query_len,
                                   'max_f_cau_len': max_f_cau_query_len,
                                   'max_f_c_num': max_f_c_querys_num,
                                   'max_doc_len': max_doc_len}



if __name__ == '__main__':
    dataset_name_list = []
    for i in range(1, 11):
        dataset_name_list.append('fold{}'.format(i))
    dataset_type_list = ['train', 'test']  #['train', 'test','valid']
    for dataset_name in dataset_name_list:
        for dataset_type in dataset_type_list:  # 对每个文件
            output_path = 'data/preprocess_test/' + dataset_name + '_' + dataset_type + '_dual.pt'
            input_path = 'data/split10/' + dataset_name + '_' + dataset_type + '.json'
            sample_list = []
            with open(input_path, 'r', encoding='utf-8') as file:
                dataset = json.load(file)
            for doc in dataset:  # 对每个文档
                doc_id = int(doc['doc_id'])
                doc_len = int(doc['doc_len'])
                doc_couples = doc['pairs']
                doc_clauses = doc['clauses']
                clause_list = []
                emotion_categorys = []
                emotion_tokens = []
                for i in range(len(doc_clauses)):
                    clause_list.append(doc_clauses[i]['clause'])
                doc_len, doc_couples, clause_list =  data_bert_trunk(doc_len, doc_couples, clause_list)
                emotion_list, cause_list = zip(*doc_couples)
                emotion_list = list(set(emotion_list))
                cause_list = list(set(cause_list))
                clause_len_list = [len(clause) for clause in clause_list]
                assert  len(clause_list) == len(clause_len_list)
                text = ''.join(clause_list)

                forward_query_list = []
                f_e_query_answer = []
                f_c_query_answer = []
                f_query_len_list = []

                # For EE
                forward_query_list.append([])
                f_query_len_list.append(0)  #the length of the question: for the same format of EE and ECE
                f_e_query_answer = [[0] * doc_len]  #emo prediction result  [[0,0,0,0...]]
                for emo_idx in emotion_list:
                    f_e_query_answer[0][emo_idx - 1] = 1

                # For ECE
                temp_emotion = set()
                for pair in doc_couples:
                    emotion_idx = pair[0]
                    cause_idx = pair[1]
                    if emotion_idx not in temp_emotion:
                        causes = []
                        for e, c in doc_couples:
                            if e == emotion_idx:
                                causes.append(c)
                        query_f = clause_list[emotion_idx - 1]
                        forward_query_list.append([query_f])
                        f_query_len_list.append(len(query_f))
                        f_query2_answer = [0] * doc_len
                        for c_idx in causes:
                            f_query2_answer[c_idx - 1] = 1
                        f_c_query_answer.append(f_query2_answer)
                        temp_emotion.add(emotion_idx)

                temp_sample = dual_sample(doc_id, doc_len, text, clause_list, clause_len_list, doc_couples,
                                          f_query_len_list, forward_query_list, f_e_query_answer, f_c_query_answer)
                sample_list.append(temp_sample)
            torch.save(sample_list, output_path)
            print(output_path, ' build finish!')

    for dataset_name in dataset_name_list:
        output_path = 'data/preprocess_test/' + dataset_name + '.pt'
        train_data = torch.load('data/preprocess_test/' + dataset_name + '_train_dual.pt')
        test_data = torch.load('data/preprocess_test/' + dataset_name + '_test_dual.pt')

        train_tokenized, train_max_len = tokenized_data(train_data)
        test_tokenized, test_max_len = tokenized_data(test_data)

        train_preprocess = pre_processing(train_tokenized, train_max_len)
        test_preprocess = pre_processing(test_tokenized, test_max_len)
        print(output_path, ' finish.')
        torch.save({'train': train_preprocess, 'test': test_preprocess}, output_path)