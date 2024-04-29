import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np


class BalancedClassSampler(Sampler):
    def __init__(self, datasets, num_samples_per_class, batch_size, drop_last=False):
        # self.dataset_src = dataset_src
        # self.dataset_tgt = dataset_tgt

        self.dataset_src = datasets.dataset_src
        self.dataset_tgt = datasets.dataset_tgt
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Organize indices by class
        self.indices_src = {cls: np.where(np.array(datasets.dataset_src.labels) == cls)[0]
                            for cls in np.unique(self.dataset_src.labels)}
        self.indices_tgt = {cls: np.where(np.array(self.dataset_tgt.labels) == cls)[0]
                            for cls in np.unique(self.dataset_tgt.labels)}
        self.classes = list(set(self.indices_src.keys()).intersection(self.indices_tgt.keys()))

    def __iter__(self):
        # Shuffle class order each epoch
        class_batches = []
        batch = []
        print("HEREEEE")
        for class_id in self.classes:
            np.random.shuffle(self.indices_src[class_id])
            np.random.shuffle(self.indices_tgt[class_id])

            # Calculate the minimum number of full batches per class
            src_batches = len(self.indices_src[class_id]) // self.num_samples_per_class
            tgt_batches = len(self.indices_tgt[class_id]) // self.num_samples_per_class
            num_full_batches = min(src_batches, tgt_batches)

            if self.drop_last:
                num_full_batches = num_full_batches // (self.batch_size // (2 * self.num_samples_per_class))

            for i in range(num_full_batches):
                start_idx = i * self.num_samples_per_class
                end_idx = start_idx + self.num_samples_per_class
                class_batches.append((self.indices_src[class_id][start_idx:end_idx],
                                      self.indices_tgt[class_id][start_idx:end_idx]))

        # Shuffle batches to mix classes
        np.random.shuffle(class_batches)
        for src_batch, tgt_batch in class_batches:
            assert len(src_batch) == len(tgt_batch), "src and tgt number of examples don't match"
            # print("src: ", src_batch)
            # print("tgt_batch: ", tgt_batch)
            full_indices = [{'src_index': src_batch[i], 'tgt_index': tgt_batch[i]} for i in range(len(src_batch))]
            batch.extend(full_indices)
            if len(batch) * 2 >= self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        # Calculate the total number of batches
        total_batches = sum(
            (min(len(self.indices_src[cls]), len(self.indices_tgt[cls])) // (self.num_samples_per_class // 2))
            for cls in self.classes)
        if self.drop_last:
            total_batches = (total_batches // (self.batch_size // (2 * self.num_samples_per_class))) * (
                        self.batch_size // (2 * self.num_samples_per_class))
        return total_batches // (self.batch_size // (2 * self.num_samples_per_class))
