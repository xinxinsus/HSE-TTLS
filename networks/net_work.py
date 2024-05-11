from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE



class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pt, target):

        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (
                    1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert_encoder = BertEncoder(configs)

        self.trans_doc = clause_transformer(configs)
        self.pred_e = Pre_Predictions(configs)

        self.pred_emo = Pre_cau_Predictions(configs)

    def forward(self, query, query_mask, query_seg, query_len, seq_len, doc_len, adj, q_type):
        # shape: batch_size, max_doc_len, 1024
        doc_sents_h, query_h = self.bert_encoder(query, query_mask, query_seg, query_len, seq_len, doc_len, q_type)
        doc_sents_h = self.trans_doc(doc_sents_h, doc_len)

        if q_type == "f_emo":
            pred = self.pred_e(doc_sents_h)
        else:
            doc_sents_h = torch.cat((query_h, doc_sents_h), dim=-1)
            pred = self.pred_emo(doc_sents_h)

        return pred

    def loss_pre(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        criterion = nn.BCELoss()
        return criterion(pred, true)

    def loss_pre_focal(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        criterion = BCEFocalLoss()
        return criterion(pred, true)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.wei = nn.Linear(768, 1)

    def forward(self, hidden_states):
        # hidden_states的第一个维度是n*batch_size。所以用[:, 0]取所有句子的[CLS]的embedding
        first_token_tensor = hidden_states[:, :, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_weight = self.wei(pooled_output)
        return pooled_weight.squeeze(-1)


class BertEncoder(nn.Module):
    def __init__(self, configs):
        super(BertEncoder, self).__init__()
        hidden_size = configs.feat_dim
        self.bert = BertModel.from_pretrained(configs.bert_cache_path, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.fc = nn.Linear(768, 1)
        self.fc_query = nn.Linear(768, 1)
        self.pooler = BertPooler(configs)

        self.f_emo = torch.nn.Parameter(torch.FloatTensor([0.2, 1]))
        self.f_cau = torch.nn.Parameter(torch.FloatTensor([0.2, 1]))

        # self.wq1=nn.Linear(768, 1)
        # self.wq2=nn.Linear(768,1)
        # self.wp1=nn.Linear(768,1)
        # self.wp2=nn.Linear(768,1)
        # self.v = nn.Linear(1, 1)
        # self.vq = nn.linear(1, 1)

    def forward(self, query, query_mask, query_seg, query_len, seq_len, doc_len, q_type):
        hidden_states = self.bert(input_ids=query.to(DEVICE),
                                  attention_mask=query_mask.to(DEVICE),
                                  token_type_ids=query_seg.to(DEVICE))[2]

        if q_type == "f_emo":
            hidden_state = torch.stack((hidden_states[0], hidden_states[8]), dim=0).to(DEVICE)
            # # hidden_states = torch.stack((hidden_states[9], hidden_states[10], hidden_states[11], hidden_states[12]),
            # #                             dim=0).to(DEVICE)
            h_wei = self.pooler(hidden_state)  # 0.4.8
            ret = torch.einsum('bx,bxyj->xyj', h_wei, hidden_state).to(DEVICE)
            ret = self.f_emo[0] * ret + self.f_emo[1] * hidden_states[12]

            # ret=self.f_emo[-1]*hidden_states[-1]
            # for i in range(len(hidden_states)):
            #     ret=ret+self.f_emo[i]*hidden_states[i]
        elif q_type == "f_cau":
            hidden_state = torch.stack((hidden_states[0], hidden_states[10]), dim=0).to(DEVICE)
            # # hidden_states = torch.stack((hidden_states[9], hidden_states[10], hidden_states[11], hidden_states[12]),
            # #                             dim=0).to(DEVICE)
            h_wei = self.pooler(hidden_state)
            ret = torch.einsum('bx,bxyj->xyj', h_wei, hidden_state).to(DEVICE)

            ret = self.f_cau[0] * ret + self.f_cau[1] * hidden_states[12]
        else:
            print("error-error-error")

        hidden_states, mask_doc, query_state, mask_query = self.get_sentence_state(ret, query_len, seq_len, doc_len,
                                                                                   q_type)

        if q_type != 'f_emo':
            alpha_q = self.fc_query(query_state).squeeze(-1)  # bs, query_len
            mask_query = 1 - mask_query  # bs, max_query_len
            alpha_q.data.masked_fill_(mask_query.bool(), -9e5)
            alpha_q = F.softmax(alpha_q, dim=-1).unsqueeze(-1).repeat(1, 1, query_state.size(-1))
            query_state = torch.sum(alpha_q * query_state, dim=1)  # bs, 768
            query_state = query_state.unsqueeze(1).repeat(1, hidden_states.size(1), 1)

        alpha = self.fc(hidden_states).squeeze(-1)
        mask_doc = 1 - mask_doc  # bs, max_doc_len, max_seq_len
        alpha.data.masked_fill_(mask_doc.bool(), -9e5)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1).repeat(1, 1, 1, hidden_states.size(-1))
        hidden_states = torch.sum(alpha * hidden_states, dim=2)  # bs, max_doc_len, 768
        # print(hidden_states.shape)

        return hidden_states.to(DEVICE), query_state.to(DEVICE)

    def get_sentence_state(self, hidden_states, query_lens, seq_lens, doc_len, q_type):
        # 对问题的每个token做注意力，获得问题句子的向量表示；对文档的每个句子的token做注意力，得到每个句子的向量表示
        sentence_state_all = []
        query_state_all = []
        mask_all = []
        mask_query = []
        max_seq_len = 0
        for seq_len in seq_lens:  # 找出最长的一句话包含多少token
            for l in seq_len:
                max_seq_len = max(max_seq_len, l)
        max_doc_len = max(doc_len)  # 最长的文档包含多少句子
        max_query_len = max(query_lens)  # 最长的问句包含多少token
        for i in range(hidden_states.size(0)):  # 对每个batch
            # 对query
            query = hidden_states[i, 1: query_lens[i] + 1]
            assert query.size(0) == query_lens[i]
            if query_lens[i] < max_query_len:
                query = torch.cat([query, torch.zeros((max_query_len - query_lens[i], query.size(1))).to(DEVICE)],
                                  dim=0)
            query_state_all.append(query.unsqueeze(0))
            mask_query.append([1] * query_lens[i] + [0] * (max_query_len - query_lens[i]))
            # 对文档sentence
            mask = []
            if q_type == "f_emo":
                begin = query_lens[i] + 1  # cls
            else:
                begin = query_lens[i] + 2  # [cls], [sep]
            sentence_state = []
            for seq_len in seq_lens[i]:
                sentence = hidden_states[i, begin: begin + seq_len]
                begin += seq_len
                if sentence.size(0) < max_seq_len:
                    sentence = torch.cat([sentence, torch.zeros((max_seq_len - seq_len, sentence.size(-1))).to(DEVICE)],
                                         dim=0)
                sentence_state.append(sentence.unsqueeze(0))
                mask.append([1] * seq_len + [0] * (max_seq_len - seq_len))
            sentence_state = torch.cat(sentence_state, dim=0).to(DEVICE)
            if sentence_state.size(0) < max_doc_len:
                mask.extend([[0] * max_seq_len] * (max_doc_len - sentence_state.size(0)))
                padding = torch.zeros(
                    (max_doc_len - sentence_state.size(0), sentence_state.size(-2), sentence_state.size(-1)))
                sentence_state = torch.cat([sentence_state, padding.to(DEVICE)], dim=0)
            sentence_state_all.append(sentence_state.unsqueeze(0))
            mask_all.append(mask)
        query_state_all = torch.cat(query_state_all, dim=0).to(DEVICE)
        mask_query = torch.tensor(mask_query).to(DEVICE)
        sentence_state_all = torch.cat(sentence_state_all, dim=0).to(DEVICE)
        mask_all = torch.tensor(mask_all).to(DEVICE)
        return sentence_state_all, mask_all, query_state_all, mask_query


class clause_transformer(nn.Module):
    def __init__(self, configs):
        super(clause_transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, doc_sents_h, doc_len):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        mask = []
        for i in range(batch):
            mask.append([1] * doc_len[i] + [0] * (max_doc_len - doc_len[i]))
        mask = torch.tensor(mask).to(DEVICE)
        mask_query = 1 - mask  # bs, max_query_len
        # doc_sents_h.data.masked_fill_(mask_query.bool(), -9e5)
        doc_sents_h = torch.transpose(doc_sents_h, 0, 1)
        output = self.transformer_encoder(src=doc_sents_h, mask=None, src_key_padding_mask=mask_query.bool())
        output = F.dropout(output, 0.1, training=self.training)
        output = torch.transpose(output, 0, 1)

        return output


class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = 768
        self.out_e = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_e = torch.sigmoid(pred_e)
        return pred_e  # shape: bs ,max_doc_len


class Pre_cau_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_cau_Predictions, self).__init__()
        self.feat_dim = 1536
        self.out_e = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_e = torch.sigmoid(pred_e)
        return pred_e  # shape: bs ,max_doc_len
