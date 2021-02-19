import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional
import numpy as np
from torch.utils.data import TensorDataset
from text import Xmls

embed_size = 10
lstm_units = 8
epochs = 50
np.random.seed(0)
torch.manual_seed(0)
num_layers = 1
num_directions = 2
batch_size = 1


xml = Xmls()
text = xml.get_xml_text(xml.xmls[0])
text = xml.preprocess_text(text)
vocab = list(set(text))
vocab_size = len(vocab)
vocab_dict = {letter: index for index, letter in enumerate(vocab)}
text = [vocab_dict[letter] for letter in text]
text = np.resize(text, [15, 318])
text = torch.tensor(text)
test_dataset = TensorDataset(text, text)
train_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset)
seq_len = 318


# in that model, each batch is a sentence. context is not saved between sentences.
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size
        )
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(
            in_features=num_directions * lstm_units,
            out_features=vocab_size)

    def forward(self, input):
        h_t = torch.zeros(num_layers * num_directions, batch_size, lstm_units, dtype=torch.double)
        c_t = torch.zeros(num_layers * num_directions, batch_size, lstm_units, dtype=torch.double)
        embed = self.embed(input.long())
        h_t, c_t = self.lstm(embed, (h_t, c_t))
        output = self.linear(h_t)
        output = torch.reshape(output, [batch_size, vocab_size, seq_len])
        return output


class Data:
    # # simple hand-written data
    # vocab_size = 5
    # data = torch.tensor([
    #     [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    #     [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    # ])  # 20
    # dataset = TensorDataset(data, data)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # train_loader, test_loader = loader, loader
    # seq_len = len(data[0])  # 20
    pass


def run_model():

    model = Model()
    model.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.8)  # LBFGS Adam

    # train
    for epoch in range(epochs):
        print('STEP: ', epoch)
        for sentence_num, (x_curr, y_curr) in enumerate(train_loader):

            def closure():
                optimizer.zero_grad()
                out = model(x_curr)
                curr_loss = criterion(out, y_curr.long())
                print("step {} sentence {} loss:".format(epoch, sentence_num), curr_loss.item())
                curr_loss.backward()
                return curr_loss

            optimizer.step(closure)

    # predict
    with torch.no_grad():
        for sentence_num, (x_curr, y_curr) in enumerate(test_loader):
            pred = model(x_curr)
            loss = criterion(pred, y_curr.long())
            soft_max = functional.softmax(pred, 1)
            res = torch.argmax(soft_max, dim=1)
            print('sentence {} test loss:'.format(sentence_num), loss.item())
            # print('inp: {}\nout: {}'.format(x_curr.numpy(), res.numpy()))
            # print([index
            #        for index, (first, second)
            #        in enumerate(zip(x_curr.numpy(), res.numpy()))
            #        if first != second])
            print("number of different indexes :")
            print(np.count_nonzero(x_curr.numpy() - res.numpy()))


run_model()
