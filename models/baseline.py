import torch
import torch.nn as nn
# import clip
import numpy as np
from models.clip_model.clip_model import ClipImageModel, ClipTextModel
from models.slotAttention import SlotAttention
from models.utils import SoftPositionEmbed

from torchvision import models


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        # clip_model, preprocess = clip.load("RN50", device=device)
        # self.visual = clip_model.visual

        self.visual = models.resnet50(pretrained=True)
        self.visual.fc = nn.Identity()
        self.visual.fc == nn.Linear(2048, 150)
        self.visual = self.visual.cuda()

    def forward(self, x, labels):
        x = x.cuda()
        img_feautes = self.visual(x)
        ce_loss_fn = nn.CrossEntropyLoss()
        return None, None, ce_loss_fn(img_feautes, labels)
