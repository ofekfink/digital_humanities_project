from torch.utils.data import Dataset
import torch


class TextFilesDS(Dataset):

    def __init__(self, text_file, char_dict, seq_len):
        with open(text_file, 'r') as f:
            self.text = f.read()[:1000]
        self.seq_len = seq_len
        self.len = len(self.text) // self.seq_len  # TODO change if overlapping sentences
        self.char_dict = char_dict

    def __len__(self):
        return self.len

    # returns the i sentence
    def __getitem__(self, index):
        start = index * self.seq_len
        end = (index + 1) * self.seq_len
        seq = self.text[start:end]
        input = self.char_dict.encode(seq)
        target = input
        return input, target

