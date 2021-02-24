import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

vocab_size = 20
embed_size = 30
lstm_units1 = 40
lstm_units2 = 50


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTMCell(embed_size, lstm_units1)  # first = x size, second is c & h size
        self.lstm2 = nn.LSTMCell(lstm_units1, lstm_units2)
        self.linear = nn.Linear(lstm_units2, vocab_size)

    def forward(self, input):
        outputs = torch.zeros(len(input), vocab_size)
        h_t = torch.zeros(1, lstm_units1, dtype=torch.double)  # 1 is batch size
        c_t = torch.zeros(1, lstm_units1, dtype=torch.double)
        h_t2 = torch.zeros(1, lstm_units2, dtype=torch.double)
        c_t2 = torch.zeros(1, lstm_units2, dtype=torch.double)
        for i, token in enumerate(input):
            embed = self.embed(token.long()).unsqueeze(0)
            h_t, c_t = self.lstm1(embed, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs[i] = output
        return outputs


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    data = np.array(
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,
         2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    inp = torch.from_numpy(data[:-1])
    # target = torch.from_numpy(data[1:])
    target = torch.from_numpy(data[:-1])

    test_target = torch.from_numpy(np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))
    # test_inp = torch.from_numpy(np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 9, 2, 3, 4]))
    test_inp = torch.from_numpy(np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 7, 1, 2, 3, 4]))

    seq = Sequence()
    seq.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)  # try also with adam, and lr

    # begin to train
    for i in range(3):
        print('STEP: ', i)


        def closure():
            optimizer.zero_grad()
            out = seq(inp)  # .forward # out - 1 X vocab_size,
            loss = criterion(out, target.long())  # target - tokens. this is hoe crossentropy loss works in pt.
            print('step ', i, ' loss:', loss.item())
            loss.backward()
            return loss


        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = seq(test_inp)  # .forward
            loss = criterion(pred, test_target.long())
            soft_max = nn.functional.softmax(pred, 1)
            res = torch.argmax(soft_max, dim=1)
            print('test loss:', loss.item())
            print('{}\n{}'.format(test_inp.numpy(), res.numpy()))
