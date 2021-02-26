import torch
from dictionary import CharDictionary
from torch.utils.data import DataLoader
from data import TextDS
from torch.nn import functional
from configuration import criterion
from torch.optim import Adam
from numpy import count_nonzero
from dictionary import cdict, CharDictionary


docx = TextDS("FILES/DOCX_TO_TEXT/146842.txt", cdict, seq_len=100)
test_loader = DataLoader(docx)

model = torch.load("TRAINED/xml_train.pt")
optimizer = Adam(model.parameters(), lr=0.01)

# test
with torch.no_grad():

    diff_counter = 0

    for x, y in test_loader:

        pred = model(x)
        loss = criterion(pred, y.long()).squeeze()
        soft_max = functional.softmax(pred, 1)
        res = torch.argmax(soft_max, dim=1)
        print('test loss:', loss.item())
        # print('{}\n{}'.format(x.numpy(), res.numpy()))
        non_zero = count_nonzero(x.numpy() - res.numpy())
        print(non_zero)
        # if non_zero is not 0:
        #     cd = CharDictionary()
        #     original = cd.decode(x.squeeze())
        #     predicted = cd.decode(res.squeeze())
        #     print(original)
        #     print(predicted)
        diff_counter = diff_counter + non_zero

    print(diff_counter)
