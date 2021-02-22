import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from params import *

data = np.array([[
    0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
    0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
    0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
    0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
    0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
    0, 1, 2, 3, 4]])

inp = torch.from_numpy(data[:-1])
target = torch.from_numpy(data[:-1])
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
test_dataset = TensorDataset(test_input, test_target)
test_loader = DataLoader(test_dataset)

# xml = Xmls()
# text = xml.get_xml_text(xml.xmls[0])
# text = xml.preprocess_text(text)
# vocab = list(set(text))
# vocab_size = len(vocab)
# vocab_dict = {letter: index for index, letter in enumerate(vocab)}
# text = [vocab_dict[letter] for letter in text]
# text = np.resize(text, [15, 318])
# seq_len = text.shape[1]
# # sentences = [text[i:i+318]]
# text = torch.tensor(text)
# test_dataset = TensorDataset(text, text)
# train_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset)
