from hand_made_data import vocab_size
import torch
from torch import nn
from hand_made_data import train_loader, test_loader
import torch.optim as optim
from numpy import count_nonzero


class Model(nn.Module):

    embed_size = 20
    hidden_size = 30
    num_directions = 2
    num_layers = 2
    batch_size = 1

    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embed_size)
        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True)
        self.linear = nn.Linear(
            in_features=self.num_directions * self.hidden_size,
            out_features=vocab_size)

    def forward(self, input):
        h = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size, dtype=torch.double)
        c = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size, dtype=torch.double)
        embed = self.embed(input.long())
        h, c = self.lstm(embed, (h, c))
        output = self.linear(h)
        output = output.permute(0, 2, 1)
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
                print(count_nonzero(x.numpy() - res.numpy()))
