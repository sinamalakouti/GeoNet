import numpy as np
import torch
from utils import calc_coeff


def maximum_mean_discrepancies( x, y, kernel="multiscale"):
    """
    # https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook
    Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)

def train_mmd(batch, model_fe, model_cls, opt, it, device,cfg, logger, writer):

    # setting training mode
    model_fe.train()
    model_cls.train()
    opt.zero_grad()

    # get data
    (_, img_src, lbl_src), (_, img_tgt, lbl_tgt) = batch
    img_src, img_tgt, lbl_src, lbl_tgt = img_src.to(device), img_tgt.to(device), lbl_src.to(device), lbl_tgt.to(device)

    unique_cls_src = np.unique(lbl_src.to('cpu'))
    unique_lbl_tgt = np.unique(lbl_tgt.to('cpu'))
    assert len(unique_cls_src) == len(unique_lbl_tgt), "labels unique don't match"
    imfeat_src = model_fe(img_src)
    imfeat_tgt =  model_fe(img_tgt)
    output_src = model_cls(imfeat_src)

    # output_tgt = model_cls(imfeat_tgt)
    # output_src, imfeat_src = model_cls(model_fe(img_src))
    # output_tgt, imfeat_tgt = model_cls(model_fe(img_tgt))
    # feature = torch.cat((imfeat_src, imfeat_tgt), dim=0)
    # output = torch.cat((output_src, output_tgt), dim=0)
    mmd_loss = 0
    print("computing MMD")
    for cls in unique_cls_src:
        imfeat_src_filtered = imfeat_src[lbl_src==cls]
        imfeat_tgt_filtered = imfeat_tgt[lbl_tgt == cls]
        mmd_loss += maximum_mean_discrepancies(
            imfeat_src_filtered,
            imfeat_tgt_filtered,
            kernel="multiscale"
        )

    print("end MMD")

    criterion_cls = torch.nn.CrossEntropyLoss()
    mmd_loss_adjusted = (0.75 * mmd_loss)
    closs = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    # compute loss

    loss = closs + mmd_loss_adjusted

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'mmdloss {mmd_loss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=closs.item(), daloss=mmd_loss.item()
            )

        logger.info(print_str)