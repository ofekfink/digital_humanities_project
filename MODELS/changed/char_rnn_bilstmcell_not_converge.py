import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

vocab_size = 20
embed_size = 30
lstm_units1 = 40
# lstm_units2 = 50


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTMCell(embed_size, lstm_units1)  # first = x size, second is c & h size
        self.lstm2 = nn.LSTMCell(embed_size, lstm_units1)
        self.linear1 = nn.Linear(lstm_units1, vocab_size)
        self.linear2 = nn.Linear(lstm_units1, vocab_size)

    def forward(self, input):
        outputs = torch.zeros(len(input), vocab_size)
        h_t1 = torch.zeros(1, lstm_units1, dtype=torch.double)  # 1 is batch size
        c_t1 = torch.zeros(1, lstm_units1, dtype=torch.double)
        h_t2 = torch.zeros(1, lstm_units1, dtype=torch.double)
        c_t2 = torch.zeros(1, lstm_units1, dtype=torch.double)
        for index in range(len(input)):
            start = input[index]
            end = input[len(input)-index-1]
            embed1 = self.embed(start.long()).unsqueeze(0)
            embed2 = self.embed(end.long()).unsqueeze(0)
            # reverse
            h_t1, c_t1 = self.lstm1(embed1, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(embed2, (h_t2, c_t2))
            output1 = self.linear1(h_t1)
            output2 = self.linear2(h_t2)
            # addition
            output = output1 + output2
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
    for i in range(10):
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
