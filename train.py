import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, functional
from numpy import count_nonzero
from model import Model
from configuration import epochs
from data import TextFilesDS
from torch.utils.data import DataLoader
from dictionary import CharDictionary
torch.manual_seed(0)


# get model
model = Model()
model.double()

# define loss and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)  # optim.LBFGS(model.parameters(), lr=0.8, history_size=50)

# get data
char_dict = CharDictionary()
xml = TextFilesDS("FILES/HUGE_TEXTS/merged_xmls.txt", char_dict, seq_len=100)
docx = TextFilesDS("FILES/HUGE_TEXTS/merged_docxs.txt", char_dict, seq_len=100)
train_loader = DataLoader(xml)
test_loader = DataLoader(docx)

# train
for i in range(epochs):

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

# test
with torch.no_grad():

    for x, y in test_loader:

        pred = model(x)
        loss = criterion(pred, y.long()).squeeze()
        soft_max = functional.softmax(pred, 1)
        res = torch.argmax(soft_max, dim=1)
        print('test loss:', loss.item())
        # print('{}\n{}'.format(x.numpy(), res.numpy()))
        print(count_nonzero(x.numpy() - res.numpy()))
