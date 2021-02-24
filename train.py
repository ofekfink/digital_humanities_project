import torch
from model import Model
from configuration import epochs
from data import TextFilesDS
from torch.utils.data import DataLoader
from dictionary import CharDictionary
from configuration import criterion
from torch.optim import Adam
torch.manual_seed(0)


# get model
model = Model()
model.double()
optimizer = Adam(model.parameters(), lr=0.01)  # optim.LBFGS(model.parameters(), lr=0.8, history_size=50)


# get training data
char_dict = CharDictionary()
xml = TextFilesDS("FILES/HUGE_TEXTS/merged_xmls.txt", char_dict, seq_len=100)
train_loader = DataLoader(xml)

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

torch.save(model, "trained_language_model.pt")
