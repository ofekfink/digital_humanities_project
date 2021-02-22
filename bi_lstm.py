from params import *
import torch
from torch import nn
from params import hidden_size
from data import inp, target, test_input, test_target
import torch.optim as optim


num_directions = 2
num_layers = 2
batch_size = 1


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.linear = nn.Linear(num_directions * hidden_size, vocab_size)

    def forward(self, input):
        input = torch.unsqueeze(input, dim=0)
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

        def closure():
            optimizer.zero_grad()
            out = model(inp)
            curr_loss = criterion(out, target.long())
            print('step ', i, ' loss:', curr_loss.item())
            curr_loss.backward()
            return curr_loss

        optimizer.step(closure)

        with torch.no_grad():
            pred = model(test_input)
            loss = criterion(pred, test_target.long())
            soft_max = nn.functional.softmax(pred, 1)
            res = torch.argmax(soft_max, dim=1)
            print('test loss:', loss.item())
            print('{}\n{}'.format(test_input.numpy(), res.numpy()))
