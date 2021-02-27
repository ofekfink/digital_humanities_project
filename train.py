import torch
from model import Model
from configuration import epochs
from data import TextDS
from torch.utils.data import DataLoader
from configuration import criterion
from torch.optim import Adam
from dictionary import cdict
torch.manual_seed(0)


if __name__ == "__main__":

    # create dictionary from training files
    vocab_size = cdict.get_vocab_size()

    # get model
    model = Model(vocab_size)
    model.double()
    optimizer = Adam(model.parameters(), lr=0.01)

    # get training data
    xml = TextDS("FILES/SMALL_TRAINING/xml_train.txt", cdict, seq_len=100)
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

    torch.save(model, "TRAINED/xml_train.pt")
