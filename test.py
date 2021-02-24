import torch
from dictionary import CharDictionary
from torch.utils.data import DataLoader
from data import TextFilesDS
from torch.nn import functional
from configuration import criterion
from torch.optim import Adam
from numpy import count_nonzero


char_dict = CharDictionary()
docx = TextFilesDS("FILES/HUGE_TEXTS/merged_docxs.txt", char_dict, seq_len=100)
test_loader = DataLoader(docx)

model = torch.load("trained_language_model.pt")
optimizer = Adam(model.parameters(), lr=0.01)  # optim.LBFGS(model.parameters(), lr=0.8, history_size=50)

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
