import torch
from transformers import VisionTextDualEncoderModel
from transformers.models.clip.modeling_clip import CLIPOutput, CLIPVisionConfig, CLIPVisionModel

from ..losses import clip_loss


class VisionTextModel(VisionTextDualEncoderModel):

    def __init__(
        self,
        config,
        vision_model,
        text_model
    ):
        super().__init__(config=None, vision_model=vision_model, text_model=text_model)

    def _forward_vision_model(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        vision_outputs = {}
        if 'transformers' in str(type(self.vision_model)):
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            vision_outputs['pooler_output'], vision_outputs['last_hidden_state'] = self.vision_model.return_hidden(pixel_values)

        return vision_outputs

    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        """
        vision_outputs = self._forward_vision_model(
            pixel_values,
            output_attentions,
            output_hidden_states,
            return_dict
        )

        pooled_output = vision_outputs['pooler_output']  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # print(self.vision_model)

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self._forward_vision_model(
            pixel_values,
            output_attentions,
            output_hidden_states,
            return_dict
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs['pooler_output']  # pooler_output
        image_embeds = self.visual_projection(image_embeds)

        print(str(type(self.vision_model)), vision_outputs.keys(), image_embeds.shape)

        text_embeds = text_outputs[1]  # pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )