import torch
import torch.nn as nn
import clip
import numpy as np
from models.clip_model.clip_model import ClipImageModel, ClipTextModel
from models.slotAttention import SlotAttention
from models.utils import SoftPositionEmbed
from torch.nn import functional as F
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    # "EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "Geonet": "a photo of a {}.",
    # "Geonet_DG": "a photo of a {}",
    # "Geonet_DS": "in {}.",
    # "Geonet_DG_DS": "a photo of a {} in {}."
}


class ObjectCentric(nn.Module):
    def __init__(self, cfg, device, classnames):
        super().__init__()
        clip_model, preprocess = clip.load("RN50", device=device)
        self.visual = clip_model.visual
        self.clipImageEncoder = ClipImageModel(model_type="RN50", remove_pooling=True)
        clipTextEncoder = ClipTextModel(model_type="RN50", freeze_model=True)
        self.device = device
        self.dim = 2048
        self.w = 7
        self.h = 7
        self.hidden_dim = self.dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softpos = SoftPositionEmbed(2, self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.dim)
        )
        self.slot_attention = SlotAttention(self.dim, self.dim, ff_mlp=self.mlp)
        self.all_class_prompts = self.get_class_prompts(classnames)
        self.cls_score = nn.Linear(self.dim//2, len(self.all_class_prompts), bias=False)
        # with torch.no_grad():
        #     prompts = clip.tokenize(self.all_class_prompts).to(self.device)
        #     # print("our device is  ", device)
        #     # print("prompt device is ", prompts)
        #     # print("clip text encoder device is : ", self.clipTextEncoder.device)
        #     text_features = clipTextEncoder.to(device)(prompts)
        #     self.cls_score.weight.copy_(text_features)
        #     self.cls_score.requires_grad_(False)

    # def gen_contrastive_prompts(self, classnames, prompt_prefix, llm_descriptions,
    #                             countries_name=['usa', 'asia'], clip_model=None):
    #     DG_DS_prompts = {}
    #     for country in countries_name:
    #         DG_DS_prompts[country] = []
    #         for name in classnames:
    #             prefix = " a photo of " + name + " "
    #             prompts_per_class = torch.cat(
    #                 [clip_model.tokenize(prefix + p.strip('-')) for p in llm_descriptions[country][name].split("\n")]).cuda()
    #             # print("prompts_per_class ")
    #             # print(prompts_per_class.shape)
    #             tmp = clip_model.encode_text(prompts_per_class)
    #             # print("temp iss    ", tmp.shape)
    #             avg_cls_embed = torch.mean(tmp, dim=0)
    #             norm_avg_cls_embed = avg_cls_embed / avg_cls_embed.norm()
    #             DG_DS_prompts[country].append(norm_avg_cls_embed.cpu())
    #     return DG_DS_prompts
    def get_class_prompts(self, class_names, dataset_name='Geonet'):
        prompts = [CUSTOM_TEMPLATES[dataset_name].format(cls.replace("_", " ")) for cls in class_names]
        return prompts

    # def get_llm_prompt_embedding(self):
    #     llm_path = 'clean_a_photo_of_a_2023_11_03_all_country_simple_batched_text_davinci_003_dictionary_geonet_classes_to_gpt_textual_descriptions.pkl'
    #
    #     classnames_llm = ["a " + name for name in classnames]
    #     classnames = [name.replace("_", " ") for name in classnames]
    #     name_lens = [len(_tokenizer.encode(name)) for name in classnames]
    #
    #     # print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    #     clip_model_ = load_clip_to_cpu(cfg)
    #     clip_model_.cuda()
    #     # #
    #     if not cfg.DATASET.COUNTRY_ENSEMBLE:
    #         # f = './LLM_Descriptions/2023_10_20_country_simple_batched_text_davinci_003_dictionary_geonet_classes_to_gpt_textual_descriptions.pkl'  # path-to-llm-descriptions
    #         f = cfg.DATASET.DESCRIPTORS_PATH
    #         llm_descriptions = np.load(f, allow_pickle=True)
    #         all_descriptive_features = self.gen_contrastive_prompts(classnames_llm, None, llm_descriptions,
    #                                                                 clip_model=clip_model_)

    def compute_contrastive_loss(self, logits_per_image, logits_per_text, labels):
        # ce_loss = nn.CrossEntropyLoss()
        # ground_truth = torch.arange(len(logits_per_image), dtype=torch.long, device=self.device)
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        total_loss = (loss_img(logits_per_image, labels)) #/ 2
        return total_loss

    def forward(self, x, labels):
        x = x.to(self.device)
        labels = labels.to(self.device)
        # text_features = self.text_features
        img_feautes, image_pos = self.clipImageEncoder(x)
        # x_pos = self.softpos(img_feautes, image_pos)
        # slots, attn = self.slot_attention(x_pos,  img_feautes)
        slots = img_feautes
        final_img_feautres = slots.reshape(slots.shape[0], slots.shape[-1], self.w, self.h)
        final_img_feautres = self.clipImageEncoder.apply_pooling(final_img_feautres)

        final_img_feautres = final_img_feautres / final_img_feautres.norm(dim=-1, keepdim=True)
        if self.training:
            logit_scale = self.logit_scale.exp()
            # cls_scores = logit_scale * final_img_feautres @ F.normalize(self.cls_score.weight, p=2.0, dim=1).t()
            cls_scores = self.cls_score(logit_scale *final_img_feautres)
            ce_loss_fn = nn.CrossEntropyLoss()
            loss = ce_loss_fn(cls_scores, labels)
            # loss = self.compute_contrastive_loss(logits_per_image, logits_per_text, labels)
            return final_img_feautres, None, loss
        else:
            logits_per_image = self.logit_scale * final_img_feautres @ self.cls_score.weight.t()
            logits_per_text = logits_per_image.t()
            loss = self.compute_contrastive_loss(logits_per_image, logits_per_text, labels)
            return logits_per_image, logits_per_text, loss



