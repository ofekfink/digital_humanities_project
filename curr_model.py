import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

vocab_size = 10
embed_size = 20
num_units = 30

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


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size )
        self.lstm_f_1 = nn.LSTM(embed_size, num_units, batch_first=True)
        self.lstm_f_2 = nn.LSTM(num_units, num_units, batch_first=True)
        self.linear_f = nn.Linear(num_units, vocab_size)

        self.lstm_b_1 = nn.LSTM(embed_size, num_units, batch_first=True)
        self.lstm_b_2 = nn.LSTM(num_units, num_units, batch_first=True)
        self.linear_b = nn.Linear(num_units, vocab_size)

    def forward(self, input):
        L = len(input)
        h_f_1 = torch.zeros(1, 1, num_units, dtype=torch.double)
        c_f_1 = torch.zeros(1, 1, num_units, dtype=torch.double)
        h_f_2 = torch.zeros(1, 1, num_units, dtype=torch.double)
        c_f_2 = torch.zeros(1, 1, num_units, dtype=torch.double)

        h_b_1 = torch.zeros(1, 1, num_units, dtype=torch.double)
        c_b_1 = torch.zeros(1, 1, num_units, dtype=torch.double)
        h_b_2 = torch.zeros(1, 1, num_units, dtype=torch.double)
        c_b_2 = torch.zeros(1, 1, num_units, dtype=torch.double)

        embed = self.embed(input.long()).unsqueeze(0)

        h_f_1, c_f_1 = self.lstm_f_1(embed, (h_f_1, c_f_1))
        h_f_2, c_f_2 = self.lstm_f_2(h_f_1, (h_f_2, c_f_2))
        output_f = self.linear_f(h_f_2)

        h_b_1, c_b_1 = self.lstm_b_1(embed.flip(1), (h_b_1, c_b_1))
        h_b_2, c_b_2 = self.lstm_b_2(h_b_1, (h_b_2, c_b_2))
        output_b = self.linear_b(h_b_2)

        output = output_f + output_b.flip(1)

        return output.squeeze()


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    data = np.array([0,1,2,3,4,  0,1,2,3,4,  0,1,2,3,4,   0,1,2,3,4,    0,1,2,3,4,    0,1,2,3,4,   0,1,2,3,4,    0,1,2,3,4,    0,1,2,3,4,    0,1,2,3,4,    0,1,2,3,4])
    inp = torch.from_numpy(data[:-1])
    # target = torch.from_numpy(data[1:])
    target = torch.from_numpy(data[:-1])

    test_target = torch.from_numpy(np.array([0, 1, 2, 3, 4,   0, 1, 2, 3, 4,   0, 1, 2, 3, 4,    0, 1, 2, 3, 4,     0, 1, 2, 3, 4]))
    test_input  = torch.from_numpy(np.array([0, 1, 2, 3, 4,   0, 1, 2, 3, 5,   0, 1, 2, 3, 4,    0, 1, 2, 3, 4,     0, 9, 2, 3, 4]))

    seq = Sequence()
    seq.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    # begin to train
    for i in range(3):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(inp)
            loss = criterion(out, target.long())
            print('step ', i, ' loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = seq(test_input)
            loss = criterion(pred, test_target.long())
            soft_max = nn.functional.softmax(pred, 1)
            res = torch.argmax(soft_max, dim=1)
            print('test loss:', loss.item())
            print('{}\n{}'.format(test_input.numpy(), res.numpy()))



