from typing import Optional
import torch
import torch.nn as nn

from transformers import AutoModel, VisionTextDualEncoderModel

from . import register_model
from .dummy_dataclasses import VisionTextDualOutput
from ..config import ModelConfig
from ..losses import clip_loss, cosine_similarity


@register_model("vision_text_encoder")
class VisionTextEncoder(torch.nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        text_model: nn.Module,
        vision_embed_dim: int,
        text_embed_dim: int,
        projection_dim: Optional[int] = 512,
        logit_scale_init_value: Optional[float] = 2.6592,
    ):
        super(VisionTextEncoder, self).__init__()

        # Models
        self.text_model = text_model
        self.vision_model = vision_model

        # Embedding dimensions
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.projection_dim = projection_dim

        # Heads
        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        vision_model: Optional[nn.Module] = None,
        text_model: Optional[nn.Module] = None,
    ) -> "VisionTextEncoder":

        if vision_model is None:
            if model_config.vision_model.name is None:
                raise ValueError(
                    "If `vision_model` is not passed as an argument, vision model `name` should be defined in the config under `model.vision_model`"
                )
            vision_model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_config.vision_model.name
            )

        if text_model is None:
            if model_config.text_model.name is None:
                raise ValueError(
                    "If `text_model` is not passed as an argument, text model `name` should be defined in the config under `model.text_model`"
                )
            text_model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_config.text_model.name
            )

        return cls(
            vision_model=vision_model,
            text_model=text_model,
            vision_embed_dim=model_config.vision_model.embed_dim,
            text_embed_dim=model_config.text_model.embed_dim,
            projection_dim=model_config.projection_dim,
            logit_scale_init_value=model_config.logit_scale_init_value,
        )

    def _forward_vision_model(
        self,
        pixel_values: torch.Tensor,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
    ):
        if "transformers" in str(type(self.vision_model)):
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        else:
            vision_outputs = self.vision_model(pixel_values)

        return vision_outputs

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
    ):
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        """
        vision_outputs = self._forward_vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = vision_outputs["pooler_output"]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        vision_outputs = self._forward_vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs.pooler_output  # pooler_output
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]  # pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text, logits_per_image = cosine_similarity(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            logit_scale=self.logit_scale,
        )

        return VisionTextDualOutput(
            vision_pooled_embeds=image_embeds,
            text_pooled_embeds=text_embeds,
            vision_model_output=vision_outputs,
            text_model_output=text_outputs,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
        )
