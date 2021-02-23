import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from text import Xmls


class HandMadeData:

    vocab_size = 10
    seq_len = 25

    @staticmethod
    def make_data_loaders():
        data = np.array([[
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 1, 2, 3, 4]])

        inp = torch.from_numpy(data[:, :-1])
        target = torch.from_numpy(data[:, :-1])
        train_dataset = TensorDataset(inp, target)
        train_loader = DataLoader(train_dataset)
        # target = torch.from_numpy(data[1:])

        test_input = torch.from_numpy(np.array([[
            0, 1, 2, 3, 4, 0, 1, 2, 3, 5,
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 9, 2, 3, 4]]))
        test_target = torch.from_numpy(np.array([[
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 1, 2, 3, 4]]))
        test_ds = TensorDataset(test_input, test_target)
        test_loader = DataLoader(test_ds)
        return train_loader, test_loader


# file data
xml = Xmls()

# train
text1 = xml.get_xml_text(xml.xmls[0])
text1 = xml.preprocess_text(text1)

# test
text2 = xml.get_xml_text(xml.xmls[1])
text2 = xml.preprocess_text(text2)[:3930]

# make vocab
vocab = list(set(text1+text2))
vocab_size = len(vocab)
vocab_dict = {letter: index for index, letter in enumerate(vocab)}

# translate & convert to data loader
text1 = [vocab_dict[letter] for letter in text1]
text1 = np.resize(text1, [12, 393])
text1 = torch.tensor(text1)
text1 = TensorDataset(text1, text1)
train_loader = torch.utils.data.DataLoader(text1, shuffle=True)

text2 = [vocab_dict[letter] for letter in text2]
text2 = np.resize(text2, [10, 393])
text2 = torch.tensor(text2)
text2 = TensorDataset(text2, text2)
test_loader = torch.utils.data.DataLoader(text2, shuffle=True)


seq_len = 393  # text.shape[1]


