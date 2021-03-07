import torch
from torch.utils.data import DataLoader
from data import TextDS, FaultedText
from torch.nn import functional
from configuration import criterion
from torch.optim import Adam
from numpy import count_nonzero
from dictionary import cdict, CharDictionary
from tagger import Tagger

if __name__ == "__main__":

    test = TextDS("FILES/SMALL_TRAINING/xml_test.txt", cdict, seq_len=100)
    # test = TextDS("FILES/DOCX_TO_TEXT/146849.txt", cdict, seq_len=100)
    test_loader = DataLoader(test)
    model = torch.load("TRAINED/xml_train.pt")
    optimizer = Adam(model.parameters(), lr=0.01)

    # test
    with torch.no_grad():
        tagger = Tagger()
        correct = 0
        line = 0

        for x, y in test_loader:

            pred = model(x)
            loss = criterion(pred, y.long()).squeeze()
            soft_max = functional.softmax(pred, 1)
            res = torch.argmax(soft_max, dim=1)
            print('test loss:', loss.item())
            # print('{}\n{}'.format(x.numpy(), res.numpy()))
            non_zero = count_nonzero(x.numpy() - res.numpy())
            correct += len(x.numpy()[0]) - non_zero
            if non_zero is not 0:
                cd = CharDictionary()
                original = cd.decode(x.squeeze())
                predicted = cd.decode(res.squeeze())
                tagger.add_error(test.file_name, line , original, predicted)
                # print(original)
                # print(predicted)
            line += 1
        tagger.print_tree_to_file()

    accuracy = 100 * correct / len(test_loader) * 100
    print("\nAccuracy: " + str(accuracy))
