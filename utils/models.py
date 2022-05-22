import os
import sys
import math
from abc import ABC

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig, AutoModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from transformers.activations import ACT2FN
from loguru import logger

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)


class BertBiSelfAttention(nn.Module):
    """
        config:
            bi_hidden_size,
            bi_num_attention_heads,
            smi_hidden_size,
            txt_hidden_size,
            smi_attention_probs_dropout_prob,
            txt_attention_probs_dropout_prob,

    """

    def __init__(self, config, mode='normal'):
        super().__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.bi_hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.bi_num_attention_heads})"
            )
        self.cross_num = {'normal': 1, 'double-cross': 2}
        self.mode = mode
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(config.bi_hidden_size / config.bi_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_smi = nn.Linear(config.smi_hidden_size, self.all_head_size)
        self.key_smi = nn.Linear(config.smi_hidden_size, self.all_head_size)
        self.value_smi = nn.Linear(config.smi_hidden_size, self.all_head_size)

        self.dropout_smi = nn.Dropout(config.smi_attention_probs_dropout_prob)

        self.query_txt = nn.Linear(config.txt_hidden_size, self.all_head_size)
        self.key_txt = nn.Linear(config.txt_hidden_size, self.all_head_size)
        self.value_txt = nn.Linear(config.txt_hidden_size, self.all_head_size)

        self.dropout_txt = nn.Dropout(config.txt_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def cross_attention(
            self,
            hidden_states1,
            hidden_states2,
            query_fun1,
            key_fun2,
            value_fun2,
            dropout_fun1,
            attention_mask2
    ):

        query_layer1 = self.transpose_for_scores(query_fun1(hidden_states1))
        key_layer2 = self.transpose_for_scores(key_fun2(hidden_states2))
        value_layer2 = self.transpose_for_scores(value_fun2(hidden_states2))

        # Q:1, K:2, V:2
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask2 is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask2
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = dropout_fun1(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer2)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

    def forward(
            self,
            hidden_states_txt,
            hidden_states_smi,
            attention_mask_txt=None,
            attention_mask_smi=None,
            output_attentions=False,
    ):
        attention_probs_txt = None
        attention_probs_smi = None
        for _ in range(self.cross_num[self.mode]):
            context_layer_txt, attention_probs_txt = self.cross_attention(
                hidden_states_txt,
                hidden_states_smi,
                self.query_txt,
                self.key_smi,
                self.value_smi,
                self.dropout_txt,
                attention_mask_smi,
            )
            context_layer_smi, attention_probs_smi = self.cross_attention(
                hidden_states_smi,
                hidden_states_txt,
                self.query_smi,
                self.key_txt,
                self.value_txt,
                self.dropout_smi,
                attention_mask_txt,
            )
            hidden_states_txt = context_layer_txt
            hidden_states_smi = context_layer_smi

        context_layer_txt = hidden_states_txt
        context_layer_smi = hidden_states_smi

        outputs = (context_layer_txt, context_layer_smi) + ((attention_probs_txt, attention_probs_smi) if
                                                            output_attentions else tuple())
        return outputs


class BertBiSelfOutput(nn.Module):
    """
        config:
            bi_hidden_size,

            txt_hidden_size,
            txt_layer_norm_eps,
            txt_hidden_dropout_prob,

            smi_hidden_size,
            smi_layer_norm_eps,
            smi_hidden_dropout_prob,
    """
    def __init__(self, config):
        super().__init__()
        self.dense_txt = nn.Linear(config.bi_hidden_size, config.txt_hidden_size)
        self.LayerNorm_txt = nn.LayerNorm(config.txt_hidden_size, eps=config.txt_layer_norm_eps)
        self.dropout_txt = nn.Dropout(config.txt_hidden_dropout_prob)

        self.dense_smi = nn.Linear(config.bi_hidden_size, config.smi_hidden_size)
        self.LayerNorm_smi = nn.LayerNorm(config.smi_hidden_size, eps=config.smi_layer_norm_eps)
        self.dropout_smi = nn.Dropout(config.smi_hidden_dropout_prob)

    def forward(
            self,
            hidden_states_txt,
            input_tensor_txt,
            hidden_states_smi,
            input_tensor_smi,
    ):
        hidden_states_txt = self.dense_txt(hidden_states_txt)
        hidden_states_txt = self.dropout_txt(hidden_states_txt)
        hidden_states_txt = self.LayerNorm_txt(hidden_states_txt + input_tensor_txt)

        hidden_states_smi = self.dense_smi(hidden_states_smi)
        hidden_states_smi = self.dropout_smi(hidden_states_smi)
        hidden_states_smi = self.LayerNorm_smi(hidden_states_smi + input_tensor_smi)

        return hidden_states_txt, hidden_states_smi


class BertBiAttention(nn.Module):
    def __init__(self, config, mode='normal'):
        super().__init__()
        self.self = BertBiSelfAttention(config, mode=mode)
        self.output = BertBiSelfOutput(config)

    def prune_heads(self, heads):
        pass

    def forward(
            self,
            hidden_states_txt,
            hidden_states_smi,
            attention_mask_txt=None,
            attention_mask_smi=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states_txt,
            hidden_states_smi,
            attention_mask_txt,
            attention_mask_smi,
            output_attentions,
        )
        attention_output = self.output(
            self_outputs[0],
            hidden_states_txt,
            self_outputs[1],
            hidden_states_smi,
        )
        outputs = attention_output + self_outputs[2:]  # add attentions if we output them
        return outputs


class BertBiIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertBiOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def feed_forward_chunk(intermediate_fun, output_fun, attention_output):
    intermediate_output = intermediate_fun(attention_output)
    layer_output = output_fun(intermediate_output, attention_output)
    return layer_output


class BertCrossLayer(nn.Module):
    """
        config:
            txt_hidden_size,
            xt_intermediate_size,
            txt_hidden_act,
            txt_layer_norm_eps,
            txt_hidden_dropout_prob,

            smi_hidden_size,
            smi_intermediate_size,
            smi_hidden_act,
            smi_layer_norm_eps,
            smi_hidden_dropout_prob,
    """

    def __init__(self, config, mode='normal'):
        super().__init__()
        self.bi_attention = BertBiAttention(config, mode=mode)

        self.txt_intermediate = BertBiIntermediate(
            config.txt_hidden_size,
            config.txt_intermediate_size,
            config.txt_hidden_act
        )
        self.txt_output = BertBiOutput(
            config.txt_intermediate_size,
            config.txt_hidden_size,
            config.txt_layer_norm_eps,
            config.txt_hidden_dropout_prob,
        )
        self.smi_intermediate = BertBiIntermediate(
            config.smi_hidden_size,
            config.smi_intermediate_size,
            config.smi_hidden_act
        )
        self.smi_output = BertBiOutput(
            config.smi_intermediate_size,
            config.smi_hidden_size,
            config.smi_layer_norm_eps,
            config.smi_hidden_dropout_prob,
        )

    def forward(
            self,
            hidden_states_txt,
            hidden_states_smi,
            attention_mask_txt=None,
            attention_mask_smi=None,
            output_attentions=False,
    ):
        self_attention_outputs = self.bi_attention(
            hidden_states_txt,
            hidden_states_smi,
            attention_mask_txt,
            attention_mask_smi,
            output_attentions,
        )
        attention_output_txt = self_attention_outputs[0]
        attention_output_smi = self_attention_outputs[1]

        outputs = self_attention_outputs[2:]  # add self attentions if we output attention weights

        layer_output_txt = feed_forward_chunk(self.txt_intermediate, self.txt_output, attention_output_txt)
        layer_output_smi = feed_forward_chunk(self.smi_intermediate, self.smi_output, attention_output_smi)

        outputs = (layer_output_txt, layer_output_smi) + outputs

        return outputs


class BertCrossEncoder(nn.Module):
    """
        config:
            num_cross_hidden_layers
    """
    def __init__(self, config, mode='normal'):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertCrossLayer(config, mode=mode) for _ in range(config.num_cross_hidden_layers)])

    def forward(
            self,
            hidden_states_txt,
            hidden_states_smi,
            attention_mask_txt=None,
            attention_mask_smi=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + ((hidden_states_txt, hidden_states_smi),)

            layer_outputs = layer_module(
                hidden_states_txt,
                hidden_states_smi,
                attention_mask_txt,
                attention_mask_smi,
                output_attentions,
            )

            hidden_states_txt, hidden_states_smi = layer_outputs[0], layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2:],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + ((hidden_states_txt, hidden_states_smi),)

        return tuple(
            v
            for v in [
                hidden_states_txt,
                hidden_states_smi,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )


"""
    CrossEncoder Config:
        bi_hidden_size: 1024,
        bi_num_attention_heads: 16,
        
        
        smi_hidden_size,
        smi_attention_probs_dropout_prob,
        smi_layer_norm_eps,
        smi_hidden_dropout_prob,
        smi_intermediate_size,
        smi_hidden_act,
        
        txt_hidden_size,
        txt_attention_probs_dropout_prob,
        txt_layer_norm_eps,
        txt_hidden_dropout_prob,
        txt_intermediate_size,
        txt_hidden_act,
  
        num_cross_hidden_layers: 6
"""


class CrossBertForMaskedLM(BertPreTrainedModel, ABC):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, txt_encoder, smi_encoder):
        super().__init__(config)

        # self.txt_encoder = BertModel(config, add_pooling_layer=False)
        # self.smi_encoder = BertModel(config, add_pooling_layer=False)
        self.cross_encoder = BertCrossEncoder(config)

        config_txt = BertConfig(
            hidden_size=config.txt_hidden_size,
            vocab_size=config.txt_vocab_size,
            hidden_act=config.txt_hidden_act,
            layer_norm_eps=config.txt_layer_norm_eps,
        )
        self.txt_cls = BertOnlyMLMHead(config_txt)
        config_smi = BertConfig(
            hidden_size=config.smi_hidden_size,
            vocab_size=config.smi_vocab_size,
            hidden_act=config.smi_hidden_act,
            layer_norm_eps=config.smi_layer_norm_eps,
        )
        self.smi_cls = BertOnlyMLMHead(config_smi)

        # Initialize weights and apply final processing
        self.post_init()

        # self.txt_encoder = AutoModel.from_pretrained(os.path.join(base_dir, config.txt_encoder),
        #                                              add_pooling_layer=False)
        # self.smi_encoder = AutoModel.from_pretrained(os.path.join(base_dir, config.smi_encoder),
        #                                              add_pooling_layer=False)
        self.txt_encoder = txt_encoder
        self.smi_encoder = smi_encoder

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids_txt=None,
            attention_mask_txt=None,
            labels_txt=None,
            input_ids_smi=None,
            attention_mask_smi=None,
            labels_smi=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_txt = self.txt_encoder(
            input_ids_txt,
            attention_mask=attention_mask_txt,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_smi = self.smi_encoder(
            input_ids_smi,
            attention_mask=attention_mask_smi,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs = self.cross_encoder(
            hidden_states_txt=outputs_txt[0],
            hidden_states_smi=outputs_smi[0],
            attention_mask_txt=attention_mask_txt,
            attention_mask_smi=attention_mask_smi,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output_txt = outputs[0]
        sequence_output_smi = outputs[1]

        prediction_scores_txt = self.txt_cls(sequence_output_txt)
        prediction_scores_smi = self.smi_cls(sequence_output_smi)

        masked_lm_loss = None
        if labels_txt is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores_txt.view(-1, self.config.txt_vocab_size), labels_txt.view(-1))
            masked_lm_loss += loss_fct(prediction_scores_smi.view(-1, self.config_smi.vocab_size), labels_smi.view(-1))

        output = (prediction_scores_txt, prediction_scores_smi) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BaseModel(nn.Module):
    def __init__(self, _config):

        super(BaseModel, self).__init__()
        self.mode = _config.mode
        if _config.bert_dir:
            bert_dir = os.path.join(base_dir, _config.bert_dir)
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            self.bert = BertModel.from_pretrained(_config.bert_name)
        self.bert_config = self.bert.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)


class DT1Model(BaseModel):
    def __init__(self,
                 _config,
                 task_num):

        super(DT1Model, self).__init__(_config)

        # 扩充词表故需要重定义
        self.bert.pooler = None
        self.bert.resize_token_embeddings(_config.len_of_tokenizer)
        out_dims = self.bert_config.hidden_size
        # mid_linear_dims = _config.mid_linear_dims

        # 下游任务模型结构构建
        # self.mid_linear = nn.Sequential(
        #     nn.Linear(out_dims, mid_linear_dims),
        #     nn.ReLU(),
        #     nn.Dropout(_config.dropout_prob)
        # )
        # self.classifier = nn.Linear(mid_linear_dims, 1)
        self.classifier = nn.Linear(out_dims, task_num)
        self.activation = nn.Sigmoid()
        if 'train' == self.mode:
            self.criterion = nn.BCELoss(reduction='none')
        else:
            self.criterion = None

        # 模型初始化
        init_blocks = [self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        out = out.last_hidden_state[:, 0, :]  # 取cls对应的embedding
        # out = self.mid_linear(out)
        out = self.activation(self.classifier(out))

        if self.criterion:
            if out.shape != labels.shape:
                labels = labels.squeeze(1)

            # loss = self.criterion(out, labels)
            loss = self.criterion(out, (labels + 1) / 2)

            return out, loss

        return out
