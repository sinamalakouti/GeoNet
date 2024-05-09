import numpy as np
import torch

from models.objectCentric import ObjectCentric
from utils import calc_coeff


def train_objectCentric_trainer(batch_iterator, model, opt, it, device, cfg, logger, writer):
    # setting training mode
    model.train()
    model = model.to(device)
    opt.zero_grad()

    # get data
    (_, img_src, lbl_src, src_idx), (_, img_tgt, lbl_tgt, tgt_idx) = next(batch_iterator)
    img_src, img_tgt, lbl_src, lbl_tgt = img_src.to(device), img_tgt.to(device), lbl_src.to(device), lbl_tgt.to(device)

    # image_logits, text_logits, closs = model(img_src.to(device), lbl_src.to(device))
    logits = model(img_src, lbl_src)
    ce_loss = torch.nn.CrossEntropyLoss()
    closs =ce_loss(logits, lbl_src)
    # compute loss
    print("closs is", closs)

    loss = closs
    # print("loss is ", loss.detach())
    # back propagation
    print("backprop")
    loss.backward()
    opt.step()

    # curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
                    'CLoss {closs:.4f}'.format(
            it + 1, cfg['training']['iteration'],
            closs=closs.item())

        logger.info(print_str)
