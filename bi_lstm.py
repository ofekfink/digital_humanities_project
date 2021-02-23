from data import vocab_size
from params import *
import torch
from torch import nn
from params import hidden_size
from data import train_loader, test_loader
import torch.optim as optim
from numpy import count_nonzero

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.linear = nn.Linear(num_directions * hidden_size, vocab_size)

    def forward(self, input):
        h = torch.zeros(num_layers * num_directions, batch_size, hidden_size, dtype=torch.double)
        c = torch.zeros(num_layers * num_directions, batch_size, hidden_size, dtype=torch.double)
        embed = self.embed(input.long())
        h, c = self.lstm(embed, (h, c))
        output = self.linear(h)
        output = output.permute(0, 2, 1)  # TODO - dataloaders need [[]] for iteration, but then this dimensions reverse
        return output


if __name__ == '__main__':

    torch.manual_seed(0)

    model = Model()
    model.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # optim.LBFGS(model.parameters(), lr=0.8, history_size=50)

    # begin to train
    for i in range(6):
        print('STEP: ', i)

        for x, y in train_loader:
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
                pred = model(x)
                loss = criterion(pred, y.long()).squeeze()
                soft_max = nn.functional.softmax(pred, 1)
                res = torch.argmax(soft_max, dim=1)
                print('test loss:', loss.item())
                # print('{}\n{}'.format(x.numpy(), res.numpy()))
                print(count_nonzero(x.numpy()-res.numpy()))
