import numpy as np
import torch

from losses.loss import MMDLoss
from utils import calc_coeff
import torch.nn as nn


class MemoryBank(nn.Module):
    def __init__(self, num_features, num_classes, momentum=0.9):
        super(MemoryBank, self).__init__()

        self.momentum = momentum

        self.register_buffer('features_src', torch.zeros(num_classes, num_features))
        self.register_buffer('features_tgt', torch.zeros(num_classes, num_features))


def train_mmd_classWise_online(iter, mb, batch_iterator, model_fe, model_cls, opt, it, device, cfg, logger, writer):
    # setting training mode
    model_fe.train()
    model_cls.train()
    opt.zero_grad()

    model_fe = model_fe.to(device)
    model_cls = model_cls.to(device)
    # get data
    batch = next(batch_iterator)
    (_, img_src, lbl_src), (_, img_tgt, lbl_tgt) = next(batch_iterator)
    img_src, img_tgt, lbl_src, lbl_tgt = img_src.to(device), img_tgt.to(device), lbl_src.to(device), lbl_tgt.to(device)

    unique_cls_src = np.unique(lbl_src.to('cpu'))
    unique_lbl_tgt = np.unique(lbl_tgt.to('cpu'))
    # assert len(unique_cls_src) == len(unique_lbl_tgt), "labels unique don't match"
    imfeat_src = model_fe(img_src.to(device))
    imfeat_tgt = model_fe(img_tgt.to(device))
    output_src = model_cls(imfeat_src.to(device))

    mmd_loss = 0
    mmd_fn = MMDLoss()
    for cls in unique_cls_src:
        imfeat_src_filtered = imfeat_src[lbl_src == cls]

        if len(imfeat_src_filtered) > 0:
            mmd_loss = mmd_loss + mmd_fn(imfeat_src_filtered, mb.features_tgt[cls])
            mb.features_src[cls] = mb.features_src[cls] * mb.momentum + imfeat_src_filtered.mean(dim=0) * (
                        1 - mb.momentum)
    mmd_loss = mmd_loss / len(unique_cls_src)
    for cls in unique_lbl_tgt:
        imfeat_tgt_filtered = imfeat_tgt[lbl_tgt == cls]
        if len(imfeat_tgt_filtered) > 0:
            mmd_loss = mmd_loss + mmd_fn(imfeat_tgt_filtered, mb.features_src[cls])
            mb.features_tgt[cls] = mb.features_tgt[cls] * mb.momentum + imfeat_tgt_filtered.mean(dim=0) * (
                        1 - mb.momentum)
    mmd_loss = mmd_loss / len(unique_cls_src)



    if iter <= 200:
        mmd_loss = 0

    # mmd_loss = mmd_linear(imfeat_src, imfeat_tgt)
    # mmd_loss += maximum_mean_discrepancies(
    #     imfeat_src_filtered,
    #     imfeat_tgt_filtered,
    #     kernel="multiscale"
    # )

    print("end MMD")

    criterion_cls = torch.nn.CrossEntropyLoss()
    mmd_loss_adjusted = (0.2 * mmd_loss)
    closs = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    # compute loss
    print("closs is", closs)
    # print("mmd loss: mm",  mmd_loss_adjusted)
    # print("mmd loss: 22222", mmd_loss)
    print("mmd loss: ", mmd_loss_adjusted)
    loss = closs + mmd_loss_adjusted
    # print("loss is ", loss.detach())
    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
                    'x: [{curr_lr:.4g}]\t' \
                    'CLoss {closs:.4f}\t' \
                    'mmdloss {mmd_loss:.4f}'.format(
            it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
            closs=closs.item(), mmd_loss=mmd_loss.item()
        )

        logger.info(print_str)
