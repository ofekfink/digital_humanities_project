from params import *
import torch
from torch import nn
from params import hidden_size
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.linear = nn.Linear(num_directions * hidden_size, vocab_size)

    def forward(self, input):
        # input = torch.unsqueeze(input, dim=0)
        h = torch.zeros(num_layers * num_directions, batch_size, hidden_size, dtype=torch.double)
        c = torch.zeros(num_layers * num_directions, batch_size, hidden_size, dtype=torch.double)
        embed = self.embed(input.long())
        h, c = self.lstm(embed, (h, c))
        output = self.linear(h).squeeze()
        return output


if __name__ == '__main__':

    torch.manual_seed(0)

    model = Model()
    model.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    # begin to train
    for i in range(3):
        print('STEP: ', i)

        for x, y in train_loader:

            y = torch.squeeze(y, 0)

            def closure():
                optimizer.zero_grad()
                out = model(x)
                curr_loss = criterion(out, y.long())
                print('step ', i, ' loss:', curr_loss.item())
                curr_loss.backward()
                return curr_loss
            optimizer.step(closure)

        with torch.no_grad():
            for x, y in test_loader:
                y = torch.squeeze(y, 0)
                pred = model(x)
                loss = criterion(pred, y.long()).squeeze()
                soft_max = nn.functional.softmax(pred, 1)
                res = torch.argmax(soft_max, dim=1)
                print('test loss:', loss.item())
                print('{}\n{}'.format(x.numpy(), res.numpy()))
