import pickle, json, decimal, math
import torch


def to_np(x):
    return x.data.cpu().numpy()


def logistic(x):
    return 1 / (1 + math.exp(-x))

# def cal_metric_dia(pred_emo_f, true_emo, pred_cau_f, true_cau, pred_pair_f, true_pairs, doc_len):
#     tp_e, fp_e, fn_e = 0, 0, 0
#     tp_c, fp_c, fn_c = 0, 0, 0
#     tp_p, fp_p, fn_p = 0, 0, 0
#     tp_neg, fp_neg, fn_neg = 0, 0, 0
#
#     true_acc=[0]*doc_len
#     pred_acc=[0]*doc_len
#     for i in true_emo:
#         true_acc[i-1]=1
#     for i in pred_emo_f:
#         pred_acc[i - 1] = 1
#
#     count = sum([1 for x, y in zip(pred_acc, true_acc) if x == y])
#
#     neg_list=[]
#     for x_emo in range(1, doc_len + 1):
#         for i in range(1, doc_len + 1):
#             if i not in true_cau:
#                 neg_list.append([x_emo,i])
#     neg_pred=[]
#
#     for x_emo in range(1, doc_len + 1):
#         for i in range(1, doc_len + 1):
#             neg_pred.append([x_emo, i])
#
#     # try:
#     for i in pred_pair_f:
#         neg_pred.remove(i)
#     # except:
#     #     None
#
#     for pred_pair in neg_pred:
#         if pred_pair in neg_list:
#             tp_neg+= 1
#         else:
#             fp_neg += 1
#     for true_pair in true_pairs:
#         if true_pair not in pred_pair_f:
#             fn_neg += 1
#     for i in range(1, doc_len + 1):
#         if i in pred_emo_f and i in true_emo:
#             tp_e += 1
#         elif i in pred_emo_f and i not in true_emo:
#             fp_e += 1
#         elif i not in pred_emo_f and i in true_emo:
#             fn_e += 1
#         if i in pred_cau_f and i in true_cau:
#             tp_c += 1
#         elif i in pred_cau_f and i not in true_cau:
#             fp_c += 1
#         elif i not in pred_cau_f and i in true_cau:
#             fn_c += 1
#     for pred_pair in pred_pair_f:
#         if pred_pair in true_pairs:
#             tp_p += 1
#         else:
#             fp_p += 1
#     for true_pair in true_pairs:
#         if true_pair not in pred_pair_f:
#             fn_p += 1
#     return [tp_e, fp_e, fn_e], [tp_c, fp_c, fn_c], [tp_p, fp_p, fn_p],[tp_neg,fp_neg,fn_neg],[count,doc_len]
#
# def cal_metric_emo(pred_emo_f, true_emo, doc_len):
#     tp_e, fp_e, fn_e = 0, 0, 0
#     tp_c, fp_c, fn_c = 0, 0, 0
#     tp_p, fp_p, fn_p = 0, 0, 0
#     for i in range(1, doc_len + 1):
#         if i in pred_emo_f and i in true_emo:
#             tp_e += 1
#         elif i in pred_emo_f and i not in true_emo:
#             fp_e += 1
#         elif i not in pred_emo_f and i in true_emo:
#             fn_e += 1
#     #     if i in pred_cau_f and i in true_cau:
#     #         tp_c += 1
#     #     elif i in pred_cau_f and i not in true_cau:
#     #         fp_c += 1
#     #     elif i not in pred_cau_f and i in true_cau:
#     #         fn_c += 1
#     # for pred_pair in pred_pair_f:
#     #     if pred_pair in true_pairs:
#     #         tp_p += 1
#     #     else:
#     #         fp_p += 1
#     # for true_pair in true_pairs:
#     #     if true_pair not in pred_pair_f:
#     #         fn_p += 1
#     return [tp_e, fp_e, fn_e]#, [tp_c, fp_c, fn_c], [tp_p, fp_p, fn_p]


def cal_metric(pred_emo_f, true_emo, pred_cau_f, true_cau, pred_pair_f, true_pairs, doc_len):
    tp_e, fp_e, fn_e = 0, 0, 0
    tp_c, fp_c, fn_c = 0, 0, 0
    tp_p, fp_p, fn_p = 0, 0, 0
    for i in range(1, doc_len + 1):
        if i in pred_emo_f and i in true_emo:
            tp_e += 1
        elif i in pred_emo_f and i not in true_emo:
            fp_e += 1
        elif i not in pred_emo_f and i in true_emo:
            fn_e += 1
        if i in pred_cau_f and i in true_cau:
            tp_c += 1
        elif i in pred_cau_f and i not in true_cau:
            fp_c += 1
        elif i not in pred_cau_f and i in true_cau:
            fn_c += 1
    for pred_pair in pred_pair_f:
        if pred_pair in true_pairs:
            tp_p += 1
        else:
            fp_p += 1
    for true_pair in true_pairs:
        if true_pair not in pred_pair_f:
            fn_p += 1
    return [tp_e, fp_e, fn_e], [tp_c, fp_c, fn_c], [tp_p, fp_p, fn_p]




def filter_unpaired(start_prob, end_prob, start, end):
    filtered_start = []
    filtered_end = []
    filtered_prob = []
    if len(start) > 0 and len(end) > 0:
        length = start[-1] + 1 if start[-1]>=end[-1] else end[-1] + 1
        temp_seq = [0] * length
        for s in start:
            temp_seq[s] += 1
        for e in end:
            temp_seq[e] += 2
        last_start = -1
        for idx in range(len(temp_seq)):
            assert temp_seq[idx]<4
            if temp_seq[idx] == 1:
                last_start = idx
            elif temp_seq[idx] == 2:
                if last_start!=-1 and idx-last_start<99:
                    filtered_start.append(last_start)
                    filtered_end.append(idx)
                    prob = start_prob[start.index(last_start)] * end_prob[end.index(idx)]
                    filtered_prob.append(prob)
                last_start = -1
            elif temp_seq[idx] == 3:
                filtered_start.append(idx)
                filtered_end.append(idx)
                prob = start_prob[start.index(idx)] * end_prob[end.index(idx)]
                filtered_prob.append(prob)
                last_start = -1
    return filtered_start, filtered_end, filtered_prob



def eval_func(all_emo):
    precision_e = all_emo[0] / (all_emo[0] + all_emo[1] + 1e-6)
    recall_e = all_emo[0] / (all_emo[0] + all_emo[2] + 1e-6)
    f1_e = 2 * precision_e * recall_e / (precision_e + recall_e + 1e-6)
    return [f1_e, precision_e, recall_e]


def float_n(value, n='0.0000'):
    value = decimal.Decimal(str(value)).quantize(decimal.Decimal(n))
    return float(value)


def write_b(b, b_path):
    with open(b_path, 'wb') as fw:
        pickle.dump(b, fw)


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js
