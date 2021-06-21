import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm

from transformers.activations import ACT2FN
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertSelfOutput, BertPredictionHeadTransform

""" Dropout Module"""


# Before using dropout, call set_seq_dropout and set_my_dropout_prob first in the __init__() of your own model
def set_seq_dropout(option):  # option = True or False
    global do_seq_dropout
    do_seq_dropout = option


def set_my_dropout_prob(p):  # p between 0 to 1
    global my_dropout_p
    my_dropout_p = p


def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training is False or p == 0:
        return x
    dropout_mask = 1.0 / (1 - p) * torch.bernoulli((1 - p)
                                                   * (x.new_zeros(x.size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3:  # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


identity_fn = lambda x: x


class LinearSelfAttnAllennlp(nn.Module):
    """
    This module use allennlp.nn.utils.masked_softmax to avoid NAN while all values are masked.
    The input mask is 1 for masked value and 0 for true value.
    """

    def __init__(self, input_size):
        super(LinearSelfAttnAllennlp, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        x = dropout(x, p=my_dropout_p, training=self.training)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        # alpha = util.masked_softmax(scores, 1 - x_mask, dim=1)
        # alpha = masked_softmax(scores, 1 - x_mask, dim=1)

        x_mask = x_mask.to(dtype=scores.dtype)
        alpha = torch.softmax(scores + x_mask * -10000.0, dim=-1)
        return alpha


class MLPWithLayerNorm(nn.Module):
    def __init__(self, config: BertConfig, input_size):
        super(MLPWithLayerNorm, self).__init__()
        self.config = config

        self.linear1 = nn.Linear(input_size, config.hidden_size)
        self.non_lin1 = ACT2FN[config.hidden_act] if isinstance(
            config.hidden_act, str) else config.hidden_act
        self.aLayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.non_lin2 = ACT2FN[config.hidden_act] if isinstance(
            config.hidden_act, str) else config.hidden_act
        self.bLayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden):
        return self.bLayerNorm(self.non_lin2(self.linear2(self.aLayerNorm(self.non_lin1(self.linear1(hidden))))))


class MHAToken(nn.Module):
    def __init__(self, config: BertConfig, bias=False):
        super().__init__()

        self.config = config

        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // self.head_num
        self.all_dim = self.head_num * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_dim, bias=bias)
        self.kv = nn.Linear(config.hidden_size, self.all_dim, bias=bias)

        self.output = BertSelfOutput(config)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.scale = 1 / (self.head_dim ** 0.5)

    def forward(self, ini_q, ini_k, mask, residual=True):
        r"""
        q: [batch, q_num, h]
        k: [batch, sent_num, token_num, h]
        mask: [batch, sent_num, token_num]

        return: [batch, q_num, sent_num, h]
        """
        q_num = ini_q.size(1)
        batch, s_num, t_num, _ = ini_k.size()

        q = self.query(ini_q).view(batch, q_num, self.head_num, self.head_dim)
        k = v = self.kv(ini_k).view(batch, s_num, t_num,
                                    self.head_num, self.head_dim)

        scores = torch.einsum("bqhd,bsthd->bhqst", q, k) * self.scale

        mask = mask.view(batch, 1, 1, s_num, t_num)

        scores = scores + mask * -10000.0

        alpha = self.dropout(torch.softmax(scores, dim=4))

        res = torch.einsum("bhqst,bsthd->bqshd", alpha,
                           v).reshape(batch, q_num, s_num, self.all_dim)

        if residual:
            output = self.output(res, ini_q[:, :, None, :])
        else:
            output = self.output(res, ini_q.new_zeros(batch, q_num, s_num, self.all_dim))

        return output


class MHATokenNoLayerNorm(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // self.head_num
        self.all_dim = self.head_num * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_dim, bias=False)
        self.kv = nn.Linear(config.hidden_size, self.all_dim, bias=False)

        self.output = nn.Linear(self.all_dim, config.hidden_size)

        self.a_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.o_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.scale = 1 / (self.head_dim ** 0.5)

    def forward(self, ini_q, ini_k, mask):
        r"""
        q: [batch, q_num, h]
        k: [batch, sent_num, token_num, h]
        mask: [batch, sent_num, token_num]

        return: [batch, q_num, sent_num, h]
        """
        q_num = ini_q.size(1)
        batch, s_num, t_num, _ = ini_k.size()

        q = self.query(ini_q).view(batch, q_num, self.head_num, self.head_dim)
        k = v = self.kv(ini_k).view(batch, s_num, t_num,
                                    self.head_num, self.head_dim)

        scores = torch.einsum("bqhd,bsthd->bhqst", q, k) * self.scale

        mask = mask.view(batch, 1, 1, s_num, t_num)

        scores = scores + mask * -10000.0

        alpha = self.a_dropout(torch.softmax(scores, dim=4))

        res = torch.einsum("bhqst,bsthd->bqshd", alpha,
                           v).reshape(batch, q_num, s_num, self.all_dim)

        output = self.o_dropout(self.output(res))

        return output


class MultiHeadTokenAttention(nn.Module):
    def __init__(self, config: BertConfig, attn_dropout_p: float, dropout_p: float):
        super().__init__()

        self.config = config

        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // self.head_num
        self.all_dim = self.head_num * self.head_dim

        self.q = nn.Linear(config.hidden_size, self.all_dim)
        self.k = nn.Linear(config.hidden_size, self.all_dim)
        self.v = nn.Linear(config.hidden_size, self.all_dim)

        self.o = nn.Linear(self.all_dim, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

        self.attention_dropout = nn.Dropout(attn_dropout_p)
        self.dropout = nn.Dropout(dropout_p)

        self.scale = 1 / (self.head_dim ** 0.5)

    def forward(self, ini_q, ini_k, mask, residual=False):
        r"""
        q: [batch, q_num, h]
        k: [batch, sent_num, token_num, h]
        mask: [batch, sent_num, token_num]

        return: [batch, q_num, sent_num, h]
        """
        q_num = ini_q.size(1)
        batch, s_num, t_num, _ = ini_k.size()

        q = self.q(ini_q).view(batch, q_num, self.head_num, self.head_dim)
        k = self.k(ini_k).view(batch, s_num, t_num, self.head_num, self.head_dim)
        v = self.v(ini_k).view(batch, s_num, t_num, self.head_num, self.head_dim)

        scores = torch.einsum("bqhd,bsthd->bhqst", q, k) * self.scale

        mask = mask.contiguous().view(batch, 1, 1, s_num, t_num)

        scores = scores + mask * -10000.0

        alpha = self.attention_dropout(torch.softmax(scores, dim=-1))

        res = torch.einsum("bhqst,bsthd->bqshd", alpha, v)
        res = res.reshape(batch, q_num, s_num, self.all_dim)
        res = self.dropout(self.o(res))

        if residual:
            res = res + ini_q.unsqueeze(2)

        res = self.layer_norm(res)

        return res


class MultiHeadAlignedTokenAttention(nn.Module):
    def __init__(self, config: BertConfig, attn_dropout_p: float, dropout_p: float):
        super().__init__()

        self.config = config

        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // self.head_num
        self.all_dim = self.head_num * self.head_dim

        self.q = nn.Linear(config.hidden_size, self.all_dim)
        self.k = nn.Linear(config.hidden_size, self.all_dim)
        self.v = nn.Linear(config.hidden_size, self.all_dim)

        self.o = nn.Linear(self.all_dim, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

        self.attention_dropout = nn.Dropout(attn_dropout_p)
        self.dropout = nn.Dropout(dropout_p)

        self.scale = 1 / (self.head_dim ** 0.5)

    def forward(self, ini_q: torch.Tensor, ini_k: torch.Tensor, mask: torch.Tensor, aligned: bool, residual=False):
        r"""
        q: [batch, q_num, h]
        k: [batch, sent_num, token_num, h]
        mask: [batch, sent_num, token_num]
        aligned: bool

        return: [batch, q_num, sent_num, h]
        """
        q_num = ini_q.size(1)
        batch, s_num, t_num, _ = ini_k.size()

        q = self.q(ini_q).view(batch, q_num, self.head_num, self.head_dim)
        k = self.k(ini_k).view(batch, s_num, t_num, self.head_num, self.head_dim)
        v = self.v(ini_k).view(batch, s_num, t_num, self.head_num, self.head_dim)

        if not aligned:
            mul_eq = "bqhd,bsthd->bhqst"
            sum_eq = "bhqst,bsthd->bqshd"
            mask_size = (batch, 1, 1, s_num, t_num)
            output_size = (batch, q_num, s_num, self.all_dim)
        else:
            assert q_num == s_num
            mul_eq = "bshd,bsthd->bhst"
            sum_eq = "bhst,bsthd->bshd"
            mask_size = (batch, 1, s_num, t_num)
            output_size = (batch, s_num, self.all_dim)

        scores = torch.einsum(mul_eq, q, k) * self.scale

        mask = mask.contiguous().view(*mask_size)

        scores = scores + mask * -10000.0

        alpha = self.attention_dropout(torch.softmax(scores, dim=-1))

        res = torch.einsum(sum_eq, alpha, v)
        res = res.reshape(*output_size)
        res = self.dropout(self.o(res))

        if residual:
            if not aligned:
                res = res + ini_q.unsqueeze(2)
            else:
                res = res + ini_q

        res = self.layer_norm(res)

        return res


class MultiHeadSelfTokenAttention(nn.Module):
    def __init__(self, config: BertConfig, attn_dropout_p: float, dropout_p: float):
        super().__init__()

        self.config = config

        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // self.head_num
        self.all_dim = self.head_num * self.head_dim

        self.q = nn.Linear(config.hidden_size, self.head_num)
        self.v = nn.Linear(config.hidden_size, self.all_dim)

        self.o = nn.Linear(self.all_dim, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.attention_dropout = nn.Dropout(attn_dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden_states, mask):
        """
        
        :param hidden_states: [batch, sent_num, token_num, h]
        :param mask: [batch, sent_num, token_num]
        """
        batch, s_num, t_num, _ = hidden_states.size()

        # [batch, sent_num, token_num, head_num]
        q = self.q(hidden_states)
        # assert q.size() == (batch, s_num, t_num, self.head_num)
        v = self.v(hidden_states).view(batch, s_num, t_num, self.head_num, self.head_dim)

        scores = q + mask.unsqueeze(-1) * -10000.0

        alpha = torch.softmax(scores, dim=2)

        alpha = self.attention_dropout(alpha)

        res = torch.einsum("bsth,bsthd->bshd", alpha, v)
        res = res.reshape(batch, s_num, self.all_dim)
        res = self.layer_norm(self.dropout(self.o(res)))

        return res


class SummaryTransformer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.mha_token = MHAToken(config)

        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_o = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_act = ACT2FN[config.hidden_act]
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, q, k, mask):
        mha_output = self.mha_token(q, k, mask, residual=False)

        ffn_output = self.ffn_o(self.ffn_act(self.ffn(mha_output)))

        hidden = self.full_layer_layer_norm(mha_output + ffn_output)

        return hidden


class SentenceReorderPredictionHead(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, query, key):
        r"""
        query: [batch, query_num, h],
        key: [batch, query_num, sentence_num, h] / [batch, sentence_num, h]
        """

        if len(key.size()) == 3:
            equation = "bqh,bsh->bqs"
        elif len(key.size()) == 4:
            equation = "bqh,bqsh->bqs"
        else:
            raise RuntimeError(query.size(), key.size())

        query = self.decoder(self.transform(query))
        scores = torch.einsum(equation, query, key)

        return scores


class SentenceReorderPredictionHeadReverse(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, query, key):
        r"""
        query: [batch, query_num, h],
        key: [batch, query_num, sentence_num, h] / [batch, sentence_num, h]
        """

        if len(key.size()) == 3:
            equation = "bqh,bsh->bqs"
        elif len(key.size()) == 4:
            equation = "bqh,bqsh->bqs"
        else:
            raise RuntimeError(query.size(), key.size())

        scores = torch.einsum(equation,
                              query, self.decoder(self.transform(key)))
        return scores


class SentenceReorderPredictionBidirectionalHead(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.transform = BertPredictionHeadTransform(config)

        self.decoder1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, query, key):
        r"""
        query: [batch, query_num, h],
        key: [batch, query_num, sentence_num, h]
        """
        transformed_query = self.transform(query)
        if len(key.size()) == 4:
            equation = "bqh,bqsh->bqs"
        else:
            equation = "bqh,bsh->bqs"
        scores1 = torch.einsum(equation,
                               self.decoder1(transformed_query), key)
        scores2 = torch.einsum(equation,
                               self.decoder2(transformed_query), key)
        return scores1, scores2


class Pooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.dense = nn.Linear(input_size, input_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooler_output = self.activation(self.dense(hidden_states))
        return pooler_output


class SentenceReorderPredictionHeadDouble(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.transform1 = BertPredictionHeadTransform(config)
        self.transform2 = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, query, key):
        r"""
        query: [batch, query_num, h],
        key: [batch, query_num, sentence_num, h]
        """
        query = self.decoder(self.transform1(query))
        key = self.transform2(key)
        return torch.einsum("bqh,bqsh->bqs", query, key)


class MaskedLMPredictionHead(nn.Module):
    def __init__(self, config: BertConfig, bert_model_embedding_weights):
        super().__init__()

        self.mlp_with_layer_norm = MLPWithLayerNorm(config, config.hidden_size * 2)

        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)

        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.mlp_with_layer_norm(hidden_states)

        target_scores = self.decoder(hidden_states) + self.bias
        return target_scores


class MaskedLMPredictionHeadPos(nn.Module):
    def __init__(self, config: BertConfig, bert_model_embedding_weights, position_embedding_size=200):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, position_embedding_size)

        self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size + position_embedding_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, hidden_states):
        seq_len = hidden_states.size(1)
        position_ids = self.position_ids[:, :seq_len]

        # (1, max_targets, dim)
        position_embeddings = self.position_embeddings(position_ids).expand(hidden_states.size(0), -1, -1)

        hidden_states = self.mlp_layer_norm(torch.cat([position_embeddings, hidden_states], dim=-1))
        # target scores : bs, max_targets, vocab_size
        target_scores = self.decoder(hidden_states) + self.bias
        return target_scores


class PositionBasedSummaryForMLM(nn.Module):
    def __init__(self, config: BertConfig, bert_model_embedding_weights, position_embedding_size=200):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, position_embedding_size)

        self.pos_emb_proj = nn.Linear(position_embedding_size, config.hidden_size)

        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, src_scores, seq_hidden, seq_mask, dropout=lambda x: x):
        # seq_len = seq_hidden.size(1)
        seq_len = src_scores.size(1)
        position_ids = self.position_ids[:, :seq_len]

        pos_h = self.pos_emb_proj(self.position_embeddings(position_ids)).expand(seq_hidden.size(0), -1, -1)

        pos_scores = pos_h.bmm(seq_hidden.transpose(1, 2))
        scores = pos_scores + src_scores
        alpha = torch.softmax(scores + seq_mask.unsqueeze(1) * -10000.0, dim=-1)

        seq_sum = dropout(alpha.bmm(seq_hidden))

        hidden_states = self.transform(seq_sum)

        target_scores = self.decoder(hidden_states) + self.bias

        return target_scores


class MultiHeadPositionBiasBasedForMLM(nn.Module):
    def __init__(self, config: BertConfig, bert_model_embedding_weights, position_embedding_size=200):
        super().__init__()
        self.config = config
        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // self.head_num

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, position_embedding_size)

        self.pos_emb_proj = nn.Linear(position_embedding_size, config.hidden_size)

        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, src_scores, seq_hidden_k, seq_hidden_v, seq_mask, dropout=lambda x: x):
        batch, _, query_len, seq_len = src_scores.size()
        
        seq_hidden_k = seq_hidden_k.view(batch, -1, self.head_num, self.head_dim)
        seq_hidden_v = seq_hidden_v.view(batch, -1, self.head_num, self.head_dim)

        position_ids = self.position_ids[:, :query_len]
        pos_q = self.pos_emb_proj(self.position_embeddings(position_ids)).expand(batch, -1, -1)
        pos_q = pos_q.view(batch, query_len, self.head_num, self.head_dim)

        pos_scores = torch.einsum("bihd,bjhd->bhij", pos_q, seq_hidden_k)
        scores = pos_scores + src_scores + seq_mask.view(batch, 1, 1, seq_len) * -10000.0
        alpha = dropout(torch.softmax(scores, dim=-1))

        seq_sum = torch.einsum("bhij,bjhd->bihd", alpha, seq_hidden_v).contiguous()
        seq_sum = seq_sum.view(batch, query_len, self.config.hidden_size)

        hidden_states = self.transform(seq_sum)

        target_scores = self.decoder(hidden_states) + self.bias

        return target_scores



class MaskedLMCopyHead(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.mlp_with_layer_norm = MLPWithLayerNorm(config, config.hidden_size * 2)

        self.decoder = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, source, source_mask):
        hidden_states = self.mlp_with_layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        scores = hidden_states.bmm(source.transpose(1, 2))
        scores = scores + source_mask.unsqueeze(1) * -10000.0

        return scores


def simple_circle_loss(output, target, mask, num_classes, softplus: torch.nn.Softplus):
    """
    output: (B, N)
    target: (B, X)
    mask: (B, N)
    num_classes = N

    loss = log(1 + \sum_i exp(s^{neg}_i) \sum_j exp(s^{pos}_j))

    """
    output = output.float()

    # seq_len = output.size(1)
    target_mask = (target == -1)
    target[target_mask] = num_classes

    # flat_target = torch.zeros((output.size(0), seq_len + 1), dtype=torch.long, device=output.device)
    # flat_target.scatter_(dim=1, index=target, value=1)
    # flat_target = flat_target[:, :seq_len]

    one_hot_target = F.one_hot(target, num_classes + 1)
    one_hot_target = one_hot_target.sum(dim=1)
    one_hot_target = one_hot_target[:, :-1].to(dtype=output.dtype)
    # assert one_hot_target.size(-1) == num_classes, one_hot_target.size()

    mask = mask.to(dtype=output.dtype)
    all_mask = (one_hot_target.sum(dim=-1) == 0)

    _flag = -1e12
    output = (1 - 2 * one_hot_target) * output  # Positive: -1 // Negative: +1

    # _zeros = torch.zeros((output.size(0), 1), device=output.device, dtype=output.dtype)
    # print(output)
    # print(one_hot_target)
    # print(mask)
    output_pos = output + \
                 torch.clamp_max((1 - one_hot_target) + mask, max=1.0) * _flag
    x = output_pos.max(dim=-1, keepdim=True)[0].detach()
    x = torch.relu_(x)
    # x = torch.cat([x, _zeros], dim=-1).max(dim=-1, keepdim=True)[0]
    output_pos = output_pos - x
    assert output_pos.size() == output.size(), (output_pos.size(), output.size())

    output_neg = output + \
                 torch.clamp_max(one_hot_target + mask, max=1.0) * _flag
    y = output_neg.max(dim=-1, keepdim=True)[0].detach()
    # y = torch.cat([y, _zeros], dim=-1).max(dim=-1, keepdim=True)[0]
    y = torch.relu_(y)
    output_neg = output_neg - y
    assert output_neg.size() == output.size(), (output_neg.size(), output.size())

    # log_sum_exp_pos = x + torch.logsumexp(output_pos, dim=-1)
    # log_sum_exp_neg = y + torch.logsumexp(output_neg, dim=-1)
    # loss = softplus(log_sum_exp_pos + log_sum_exp_neg)
    # masked_loss = loss.masked_fill(all_mask, 0.)

    sum_exp_pos = output_pos.exp().sum(dim=-1)
    # print(sum_exp_pos[all_mask])
    # sum_exp_pos[all_mask] = 1.
    # if torch.any(torch.isinf(sum_exp_pos)):
    #     print("sum_exp_pos")
    #     print(sum_exp_pos)
    # print(sum_exp_pos)
    sum_exp_neg = output_neg.exp().sum(dim=-1)
    # print(sum_exp_neg[all_mask])
    # if torch.any(torch.isinf(sum_exp_neg)):
    #     print("sum_exp_neg")
    #     print(sum_exp_neg)
    # print(sum_exp_neg)
    # gamma = (x + y).exp().squeeze(-1)
    t = -(x + y).squeeze(-1)
    assert t.size() == sum_exp_pos.size() == all_mask.size(
    ), (t.size(), sum_exp_pos.size(), all_mask.size())
    loss = (t.exp() + sum_exp_pos * sum_exp_neg + 1e-8).log() - t
    # Remove the scalar?
    # loss = (t.exp() + sum_exp_pos * sum_exp_neg + 1e-8).log()

    # print(loss[all_mask])
    # if torch.any(torch.isinf(loss)):
    #     print("loss")
    #     print(loss)
    # print(loss)
    masked_loss = loss.masked_fill(all_mask, 0.)
    # if torch.any(torch.isinf(masked_loss)):
    #     print("masked_loss")
    #     print(masked_loss)
    # print(masked_loss)

    # print(output_pos.exp())
    # sum_exp_pos = output_pos.exp().sum(dim=-1)
    # print(sum_exp_pos)
    # sum_exp_pos[all_mask] = 0.
    # print(output_neg.exp())
    # sum_exp_neg = output_neg.exp().sum(dim=-1)
    # print(sum_exp_neg)
    # sum_exp_neg[all_mask] = 0.
    # loss = (1 + sum_exp_pos * sum_exp_neg).log()
    # print(loss)
    # masked_loss = loss.masked_fill(all_mask, 0.)

    # exp_pos = torch.logsumexp(output_pos, dim=-1)
    # exp_pos = exp_pos.masked_fill(all_mask, _flag)
    # print("exp_pos", exp_pos)
    # exp_neg = torch.logsumexp(output_neg, dim=-1)
    # exp_neg = exp_neg.masked_fill(all_mask, _flag)
    # print("exp_neg", exp_neg)
    # loss = softplus(exp_pos + exp_neg)
    # print("loss", loss)
    # masked_loss = loss.masked_fill(all_mask, 0.)
    # print("softplus", masked_loss)
    # # masked_loss = masked_loss.sum()
    # masked_loss = masked_loss.mean()

    # print(masked_loss.sum())

    # valid_num = (1 - all_mask.to(dtype=masked_loss.dtype)).sum().item()

    return masked_loss.sum()


def fix_circle_loss(output, target, mask, num_classes, softplus: torch.nn.Softplus):
    """
    output: (B, N)
    target: (B, X)
    mask: (B, N)
    num_classes = N

    loss = log(1 + \sum_i exp(s^{neg}_i) \sum_j exp(s^{pos}_j))

    """
    output = output.float()

    # seq_len = output.size(1)
    target_mask = (target == -1)
    target[target_mask] = num_classes

    one_hot_target = F.one_hot(target, num_classes + 1)
    one_hot_target = one_hot_target.sum(dim=1)
    one_hot_target = one_hot_target[:, :-1].to(dtype=output.dtype)

    mask = mask.to(dtype=output.dtype)
    all_mask = (one_hot_target.sum(dim=-1) == 0)

    _flag = -1e12

    ap = torch.clamp_min(1 - output.detach(), min=0.)
    an = torch.clamp_min(output.detach(), min=0.)

    delta_p = 1
    delta_n = 0

    output = (1 - 2 * one_hot_target) * output  # Positive: -1 // Negative: +1

    output_pos = ap * (delta_p + output) + \
                 torch.clamp_max((1 - one_hot_target) + mask, max=1.0) * _flag

    x = output_pos.max(dim=-1, keepdim=True)[0].detach()
    x = torch.relu_(x)

    output_pos = output_pos - x
    assert output_pos.size() == output.size(), (output_pos.size(), output.size())

    output_neg = an * (output - delta_n) + \
                 torch.clamp_max(one_hot_target + mask, max=1.0) * _flag

    y = output_neg.max(dim=-1, keepdim=True)[0].detach()
    y = torch.relu_(y)

    output_neg = output_neg - y
    assert output_neg.size() == output.size(), (output_neg.size(), output.size())

    sum_exp_pos = output_pos.exp().sum(dim=-1)

    sum_exp_neg = output_neg.exp().sum(dim=-1)

    t = -(x + y).squeeze(-1)
    assert t.size() == sum_exp_pos.size() == all_mask.size(
    ), (t.size(), sum_exp_pos.size(), all_mask.size())
    loss = (t.exp() + sum_exp_pos * sum_exp_neg + 1e-8).log() - t

    masked_loss = loss.masked_fill(all_mask, 0.)

    return masked_loss


def fixed_circle_loss_clean(output, target, mask, num_classes, softplus: torch.nn.Softplus):
    """
    output: (B, N)
    target: (B, X)
    mask: (B, N)
    num_classes = N

    loss = log(1 + \sum_i exp(s^{neg}_i) \sum_j exp(s^{pos}_j))

    \gamma = 1, m = 0

    """
    output = output.float()

    # seq_len = output.size(1)
    target_mask = (target == -1)
    target[target_mask] = num_classes

    one_hot_target = F.one_hot(target, num_classes + 1)
    one_hot_target = one_hot_target.sum(dim=1)
    one_hot_target = one_hot_target[:, :-1].to(dtype=output.dtype)

    mask = mask.to(dtype=output.dtype)
    all_mask = (one_hot_target.sum(dim=-1) == 0)

    mask_for_pos = torch.clamp_max(1 - one_hot_target + mask, max=1.0)
    mask_for_neg = torch.clamp_max(one_hot_target + mask, max=1.0)

    _flag = -1e12

    ap = torch.clamp_min(1 - output.detach(), min=0.)
    an = torch.clamp_min(output.detach(), min=0.)

    delta_p = 1
    delta_n = 0

    output = (1 - 2 * one_hot_target) * output  # Positive: -1 // Negative: +1

    logit_pos = ap * (delta_p + output) + mask_for_pos * _flag

    x = logit_pos.max(dim=-1, keepdim=True)[0].detach()
    x = torch.relu_(x)

    logit_pos = logit_pos - x
    # assert output_pos.size() == output.size(), (output_pos.size(), output.size())

    logit_neg = an * (output - delta_n) + mask_for_neg * _flag

    y = logit_neg.max(dim=-1, keepdim=True)[0].detach()
    y = torch.relu_(y)

    logit_neg = logit_neg - y

    loss = softplus(x + y + torch.logsumexp(logit_pos, dim=-1) + torch.logsumexp(logit_neg, dim=-1))

    masked_loss = loss.masked_fill(all_mask, 0.)

    return masked_loss


def weighted_sum(q, kv, mask, v=None, _dropout=identity_fn):
    r"""
    q: [batch, h]
    kv: [batch, s, h]
    mask: [batch, s],
    v: [batch, s, h]
    dropout: nn.Dropout
    """
    if v is None:
        v = kv

    scores = torch.einsum("bh,bsh->bs", q, kv)
    scores = scores + mask * -10000.0
    alpha = torch.softmax(scores, dim=-1)
    res = torch.einsum("bs,bsh->bh", _dropout(alpha), v)
    return res, scores


def mul_weighted_sum(q, kv, mask, v=None, _dropout=identity_fn):
    r"""
    q: [batch, q, h]
    kv: [batch, s, h]
    mask: [batch, s]
    """
    if v is None:
        v = kv

    scores = torch.einsum("bqh,bsh->bqs", q, kv)
    scores = scores + mask.unsqueeze(1) * -10000.0
    alpha = _dropout(scores.softmax(dim=-1))
    res = torch.einsum("bqs,bsh->bqh", alpha, v)
    return res, scores


def sentence_sum(q, kv, mask, v=None, _dropout=identity_fn):
    r"""
    q: [batch, h]
    kv: [batch, s, t, h]
    mask: [batch, s, t]
    """
    if v is None:
        v = kv

    scores = torch.einsum('bh,bsth->bst', q, kv)
    scores = scores + mask * -10000.0
    alpha = _dropout(scores.softmax(dim=-1))
    res = torch.einsum('bst,bsth->bsh', alpha, v)
    return res, scores


def mul_sentence_sum(q, kv, mask, v=None, _dropout=identity_fn):
    r"""
    q: [batch, x, h]
    kv: [batch, s, t, h]
    mask: [batch, s, t]
    """
    if v is None:
        v = kv

    scores = torch.einsum("bqh,bsth->bqst", q, kv)
    scores = scores + mask.unsqueeze(1) * -10000.0
    alpha = _dropout(scores.softmax(dim=3))
    res = torch.einsum("bqst,bsth->bqsh", alpha, v)
    return res, scores


def mul_score(q, k, mask, v=None):
    r"""
    q: [batch, q, h]
    k: [batch, q, s, h]
    mask: [batch, s]
    """
    if v is None:
        v = k

    scores = torch.einsum("bqh,bqsh->bqs", q, k)
    scores = scores + mask.unsqueeze(1) * -10000.0
    alpha = scores.softmax(dim=2)
    res = torch.einsum("bqs,bqsh->bqh", alpha, v)
    return res, scores


def multi_head_sent_sum(q, k, v, mask, head_num, attn_dropout=identity_fn):
    r"""
    :param q: [batch, q_num, h]
    :param k: [batch, s_num, t_num, h]
    :param v: [batch, s_num, t_num, h]
    :param mask: [batch, s_num, t_num]
    :param head_num: int
    :param attn_dropout: nn.Dropout
    :return:
    """
    b, x, h = q.size()
    s, t = k.size(1), k.size(2)
    head_dim = h // head_num
    q = q.view(b, x, head_num, head_dim)
    k = k.view(b, s, t, head_num, head_dim)
    v = v.view(b, s, t, head_num, head_dim)
    scores = torch.einsum("bqhd,bsthd->bhqst", q, k)
    alpha = attn_dropout(torch.softmax(scores + mask.view(b, 1, 1, s, t) * -10000.0, dim=-1))
    res = torch.einsum("bhqst,bsthd->bqshd", alpha, v).reshape(b, x, s, h)
    return res


def multi_head_sum(q, k, v, mask, head_num, attn_dropout=identity_fn, return_scores=False):
    r"""
    :param q: [batch, q_num, h]
    :param k: [batch, t_num h]
    :param v: [batch, t_num, h]
    :param mask: [batch, t_num]
    :param head_num: int
    :param attn_dropout: nn.Dropout
    :param return_scores: bool
    :return:
    """
    b, x, h = q.size()
    t = k.size(1)
    head_dim = h // head_num
    q = q.view(b, x, head_num, head_dim)
    k = k.view(b, t, head_num, head_dim)
    v = v.view(b, t, head_num, head_dim)
    scores = torch.einsum("bqhd,bthd->bhqt", q, k)
    
    if return_scores:
        return scores
    
    alpha = attn_dropout(torch.softmax(scores + mask.view(b, 1, 1, t) * -10000.0, dim=-1))
    res = torch.einsum("bhqt,bthd->bqhd", alpha, v).reshape(b, x, h)

    return res


def cross_entropy_loss_and_accuracy(_scores, _labels, _ignore_index=-1):
    r"""
    _scores: [batch, N, M],
    _labels: [batch, N]
    """
    # FIXME: This should be keep consistent with other loss functions
    #  which is averaged 
    loss_fct = nn.CrossEntropyLoss(ignore_index=_ignore_index)

    _loss = loss_fct(input=_scores.reshape(-1, _scores.size(-1)),
                     target=_labels.reshape(-1))

    _, _pred = _scores.max(dim=-1)
    _acc = torch.sum(_pred == _labels).to(_scores.dtype)
    _valid_num = torch.sum(_labels != _ignore_index).item()

    return _loss, _acc, _valid_num


def mask_scores_with_labels(_scores, _labels, _ignore_index=-1):
    r"""
    _scores: [batch, N, M],
    _labels: [batch, N]
    """
    _N = _scores.size(-1)
    _labels = _labels.masked_fill(_labels == _ignore_index, _N)
    # _labels[_labels == _ignore_index] = _N
    _labels_one_hot = F.one_hot(_labels, _N + 1)
    _labels_one_hot = torch.narrow(_labels_one_hot,
                                   dim=-1, start=0, length=_N)

    _scores = _scores + _labels_one_hot * -10000.0
    return _scores


def get_attention_reward(_scores, _max_k, hard=True):
    r"""
    Sample the top `_max_k` items from the categorical distribution
    defined by `_scores`. The relatvie weight is kept as the initial score ?
    """
    pass


def check_nan(tensor, msg):
    if torch.any(torch.isnan(tensor)):
        print(msg)
