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
        
        # Classification head
        # We can reuse lm_head if dimensions match, but usually they don't (vocab_size vs num_labels)
        # So we create a new head.
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights of the new head
        self.score.weight.data.normal_(mean=0.0, std=config.initializer_range)

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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Process whisper features using the parent class logic or manually
        # KimiAudioModel.forward handles whisper feature processing and then calls super().forward
        # But KimiAudioModel.forward calls super().forward which is MoonshotKimiaForCausalLM.forward
        # MoonshotKimiaForCausalLM.forward computes LM loss.
        # We want to bypass LM head.
        
        # So we should call self.model() directly, but we need to handle whisper features first.
        # Copying logic from KimiAudioModel.forward for whisper features:
        
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

        logits = self.score(pooled_logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

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