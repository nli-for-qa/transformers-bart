from typing import List, Tuple
import os
import random
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)


def shuffle_in_sync(source: List, target: List) -> Tuple[List, List]:
    temp = list(zip(source, target))
    random.shuffle(temp)
    source, target = zip(*temp)

    return list(source), list(target)


class SummarizationDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_dir="./cnn-dailymail/cnn_dm/",
                 type_path="train",
                 block_size=1024):
        super(SummarizationDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []

        print("loading " + type_path + " source.")

        with open(os.path.join(data_dir, type_path + ".source"), "r") as f:
            source_lines = f.readlines(
            )  # this will have \n. This is how hugging face code was. Don't know if it is good or bad.
        print("loading " + type_path + " target.")
        with open(os.path.join(data_dir, type_path + ".target"), "r") as f:
            target_lines = f.readlines()
            # shuffle in sync
            source_lines, target_lines = shuffle_in_sync(
                source_lines, target_lines)

            for text in source_lines:  # each text is a line and a full story
                tokenized = tokenizer.batch_encode_plus([text],
                                                        max_length=block_size,
                                                        pad_to_max_length=True,
                                                        return_tensors="pt")
                self.source.append(tokenized)

            for text in target_lines:  # each text is a line and a summary
                tokenized = tokenizer.batch_encode_plus([text],
                                                        max_length=block_size,
                                                        pad_to_max_length=True,
                                                        return_tensors="pt")
                # Let the model attend to pad tokens in the target.
                #tokenized['attention_mask'][tokenized['attention_mask']
                #                            == 0] = 1
                self.target.append(tokenized)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()

        src_mask = self.source[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        target_mask = self.target[index]["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }
