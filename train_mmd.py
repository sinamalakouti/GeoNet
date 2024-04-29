from dotdict import dotdict

from basic_trainer import BasicTrainer
import torch

import time
import argparse
import os
import yaml
import random
import shutil
import numpy as np
import torch
from torch import inverse, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loader import get_dataloader
from loader.joint_class_aware_loader import BalancedClassSampler
from models import get_model
from optimizers import get_optimizer, get_scheduler
from UDA_trainer import get_trainer, val
from losses import get_loss
from utils import cvt2normal_state, get_logger, loop_iterable, get_parameter_count
import wandb
class MMDTrainer(BasicTrainer):
    """
    MMD trainer.

    Input arguments:
        classifier: nn.Module = nn for the classification.
        feature_extractor: nn.Module = nn for feature extractor (if incorporate into the classifier set to None).
        optimizer: torch.optim.Optimizer = optimizer for training the model.
        scheduler: torch.optim.lr_scheduler.ExponentialLR = scheduler for shrinking the lr during the training.
        patience: int = number of epochs without improving before being stopped.
        epochs: int = number of epochs to be performed.
        mmd_lambda: float (.25) = parameter to regulate the tradeoff between classification and mmd loss.
        kernel: str ("multiscale") = kernel used for mmd.

    """

    def __init__(
            self,
            classifier: nn.Module,
            feature_extractor: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.ExponentialLR,
            patience: int,
            epochs: int,
            mmd_lambda: float = .25,
            kernel: str = "multiscale"
    ) -> None:

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mmd_lambda = mmd_lambda
        self.kernel = kernel

        self.extra_saves = dict(
            classifier=classifier,
            feature_extractor=feature_extractor,
            optimizer=optimizer,
            scheduler=scheduler
        )

        super().__init__(patience=patience, epochs=epochs, **self.extra_saves)

    def classifier_step(
            self,
            shrinked_feature_map_source,
            labels
    ) -> Tensor:

        outputs = self.classifier(shrinked_feature_map_source)
        classifier_loss = self.cross_entropy(outputs, labels)
        return classifier_loss

    def train(
            self,
            training_loader: DataLoader,
            testing_loader: DataLoader
    ) -> None:

        progress_bar_epoch = trange(1, self.epochs, leave=True, desc="Epoch")
        self.patienting = 0
        self.best_accuracy = 0
        self.previous_epoch_loss = 1e5

        for epoch in progress_bar_epoch:
            self.classifier.train()
            self.feature_extractor.train()
            total_classifier_loss, mmd_total_loss, n_samples = 0, 0, 0
            progress_bar_batch = tqdm(enumerate(training_loader), leave=False, total=len(training_loader),
                                      desc="Training")

            for idx, data in progress_bar_batch:
                self.optimizer.zero_grad()

                source_images = data["source_domain_image"].to(self.device, dtype=torch.float)
                target_images = data["target_domain_image"].to(self.device, dtype=torch.float)
                labels = data["source_domain_label"].to(self.device, dtype=torch.long)

                # mmd step
                shrinked_feature_map_source = self.feature_extractor(source_images)
                shrinked_feature_map_target = self.feature_extractor(target_images)

                mmd_loss = self.maximum_mean_discrepancies(
                    shrinked_feature_map_source,
                    shrinked_feature_map_target,
                    self.kernel
                )

                mmd_loss_adjusted = (self.mmd_lambda * mmd_loss)

                classification_loss = self.classifier_step(shrinked_feature_map_source, labels)

                loss = classification_loss + mmd_loss_adjusted
                loss.backward()
                self.optimizer.step()

                total_classifier_loss += classification_loss.item()
                mmd_total_loss += mmd_loss.item()
                n_samples += data["source_domain_image"].size()[0]

                loss_progress = {
                    "classifier_loss": total_classifier_loss / n_samples,
                    "mmd_loss": mmd_total_loss / n_samples,
                    "total_loss": (total_classifier_loss + mmd_total_loss) / n_samples
                }

                if (idx % 2 == 0 and idx):
                    progress_bar_batch.set_postfix(loss_progress)

            if self.scheduler:
                self.scheduler.step()

            wandb.log(loss_progress, commit=False)
            validation_metrics = self.validation(testing_loader, epoch)
            progress_bar_epoch.set_postfix(validation_metrics)

    def maximum_mean_discrepancies(self, x, y, kernel="multiscale"):
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

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

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


def train_eval():
    #################################################################################################
    config = dotdict({
        'optimizer': 'adam',
        'lr_classifier': .0005,
        'lr_backbone': .0001,
        'weight_decay': 0.02,
        'scheduler': 'exponential',
        'gamma': .96,
        'batch_size': 32,
        'backbone': 'resnet50',
        'mmd_lambda': .75,
        'kernel': 'multiscale'
    })

    print("-" * 150);
    config.print_all()

    #################################################################################################

    transform_source = T.Compose([
        T.Pad(padding=10),
        T.Resize(size=(256, 256)),
        T.RandomPerspective(distortion_scale=0.3, p=.8),
        T.ToTensor(),
    ])

    transform_target = T.Compose([T.ToTensor(), T.Resize(size=(256, 256))])
    config.transform_source = transform_source.__str__()
    config.transform_target = transform_target.__str__()

    target, source = "adaptiope_small/product_images", "adaptiope_small/real_life"
    direction = f"{source.split('/')[1][0]} to {target.split('/')[1][0]}"

    train, test = create_train_test(test_size=.2,
                                    source_domain_path=source,
                                    target_domain_path=target,
                                    transform_source=transform_source,
                                    transform_target=transform_target,
                                    batch_size=config.batch_size)

    #################################################################################################

    classifier = MMDClassifer()
    feature_extractor = BottleNeckFeatureExtractor()

    #################################################################################################

    params = [{'params': feature_extractor.parameters(), 'lr': config.lr_backbone},
              {'params': classifier.parameters(), 'lr': config.lr_classifier}]

    optimizer = torch.optim.Adam(params, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    #################################################################################################

    trial = MMDTrainer(
        feature_extractor=feature_extractor,
        classifier=classifier,
        mmd_lambda=config.mmd_lambda,
        kernel=config.kernel,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=25,
        epochs=100
    )

    print("-" * 150);
    print(classifier);
    print("-" * 150)
    print("-" * 150);
    print("dense_layer\n");
    print(feature_extractor.dense_layer);
    print("-" * 150)
    print("-" * 150);
    print("bottle_neck\n");
    print(feature_extractor.bottle_neck);
    print("-" * 150)
    # check_grad_names(feature_extractor);print("-"*150);
    check_grad_names(classifier);
    print("-" * 150);

