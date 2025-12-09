import os
import torch
from torch import nn
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging
from finetune_codes.model import KimiAudioModel
from .configuration_multiclass import MultiClassConfig

logger = logging.get_logger(__name__)

class MultiClassModel(KimiAudioModel):
    config_class = MultiClassConfig

    def __init__(self, config: MultiClassConfig):
        # Initialize parent (KimiAudioModel -> MoonshotKimiaForCausalLM)
        # This initializes self.model, self.lm_head, and self.whisper_model
        super().__init__(config)
        
        self.num_labels = config.num_labels
        self.config.num_labels = config.num_labels
        
        # Sentence Branch
        sent_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            # dim_feedforward=config.intermediate_size,
            dim_feedforward=4096,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.sent_encoder = nn.TransformerEncoder(sent_layer, num_layers=config.sent_num_layers)
        self.sent_head = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        
        # Word Branch (Cross-Attention)
        word_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            # dim_feedforward=config.intermediate_size,
            dim_feedforward=4096,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.word_decoder = nn.TransformerDecoder(word_layer, num_layers=config.word_num_layers)
        self.word_head = nn.Linear(config.hidden_size, config.word_num_labels, bias=False)
        self.word_query_embed = nn.Embedding(config.max_word_len, config.hidden_size)

        # Initialize weights
        self.sent_head.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.word_head.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.word_query_embed.weight.data.normal_(mean=0.0, std=config.initializer_range)

        # Remove unused heads to avoid DDP errors
        if hasattr(self, 'lm_head'):
            del self.lm_head
        if hasattr(self, 'mimo_output'):
            del self.mimo_output
        if hasattr(self, 'audio_detect_head'):
            del self.audio_detect_head

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        word_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if whisper_input_feature is not None and whisper_input_feature.numel() > 0:
            whisper_input_feats = whisper_input_feature.to(input_ids.device if input_ids is not None else inputs_embeds.device)
            whisper_feats = self.whisper_model(whisper_input_feats)
            whisper_feats = whisper_feats.reshape(
                whisper_feats.shape[0],
                int(whisper_feats.shape[1] // 4),
                whisper_feats.shape[2] * 4,
            )
        else:
            whisper_feats = whisper_input_feature

        # Now call the backbone (self.model is MoonshotKimiaModel)
        transformer_outputs = self.model(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        pooled_logits = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        # --- New Architecture Forward ---
        
        # 1. Shared Encoder (Backbone is the shared encoder)
        # hidden_states: [Batch, Seq_Len, Dim]
        # attention_mask needs to be converted to key_padding_mask for TransformerEncoder
        # attention_mask: 1 for valid, 0 for padding. 
        # TransformerEncoder expects src_key_padding_mask: True for padding, False for valid.
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
            
        # shared_out = self.shared_encoder(hidden_states, src_key_padding_mask=key_padding_mask)
        shared_out = hidden_states
        
        # 2. Sentence Branch
        sent_out = self.sent_encoder(shared_out, src_key_padding_mask=key_padding_mask)
        # Pooling for sentence (using the same logic as before: last token)
        sent_pooled = sent_out[torch.arange(batch_size, device=sent_out.device), sequence_lengths]
        sent_logits = self.sent_head(sent_pooled)
        
        # 3. Word Branch (Cross-Attention)
        # Construct Word Queries
        # word_labels shape: [Batch, Max_Word_Len] (if provided) or we use config.max_word_len
        current_max_word_len = self.config.max_word_len
        if word_labels is not None:
            current_max_word_len = word_labels.shape[1]
            
        # Create query embeddings: [1, Max_Word_Len, Dim] -> [Batch, Max_Word_Len, Dim]
        word_queries = self.word_query_embed.weight[:current_max_word_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-Attention: tgt=Queries, memory=Shared_Out
        # memory_key_padding_mask is the same as key_padding_mask from input
        word_out = self.word_decoder(tgt=word_queries, memory=shared_out, memory_key_padding_mask=key_padding_mask)
        word_logits = self.word_head(word_out) # [Batch, Max_Word_Len, Word_Num_Labels]

        loss = None
        if labels is not None:
            # Sentence Loss
            loss_fct = nn.CrossEntropyLoss()
            sent_loss = loss_fct(sent_logits.view(-1, self.num_labels), labels.view(-1).long())
            
            # Word Loss (if word_labels provided)
            if word_labels is not None:
                word_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                word_loss = word_loss_fct(word_logits.view(-1, self.config.word_num_labels), word_labels.view(-1).long())
                loss = 0.5 * sent_loss + 0.5 * word_loss
            else:
                loss = sent_loss

        # For compatibility, return sent_logits as main logits
        # logits = sent_logits
        logits = (sent_logits, word_logits)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )