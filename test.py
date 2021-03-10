import torch
from torch.utils.data import DataLoader
from data import TextDS
from torch.nn import functional
from configuration import criterion
from torch.optim import Adam
from numpy import count_nonzero
from dictionary import cdict
from tagger import Tagger
from printer import Printer

if __name__ == "__main__":

    # tagger = Tagger()
    printer = Printer()
    model = torch.load("TRAINED/xml_train.pt")
    optimizer = Adam(model.parameters(), lr=0.01)

    # ===================== TEST XMLS =====================
    '''
    xml_test = TextDS("FILES/SMALL_TRAINING/xml_test.txt", cdict, seq_len=100)
    test_loader = DataLoader(xml_test)

    with torch.no_grad():

        correct = 0
        seq_num = 0

        for x, y in test_loader:

            pred = model(x)
            loss = criterion(pred, y.long()).squeeze()
            soft_max = functional.softmax(pred, 1)
            res = torch.argmax(soft_max, dim=1)
            print('test loss:', loss.item())
            non_zero = count_nonzero(x.numpy() - res.numpy())
            seq_len = len(x.numpy()[0])
            correct = correct + seq_len - non_zero

    loader_len = len(test_loader)
    accuracy = (100 * correct) / (loader_len * 100)  # 100 for percentage, and 100 for sequence length
    print("xmls testing")
    print("\nAccuracy: " + str(accuracy))
    '''
    # ===================== TEST DOCX =====================

    docx_test = TextDS("FILES/DOCX_TO_TEXT/146844.txt", cdict, seq_len=100)
    docx_test_loader = DataLoader(docx_test)

    with torch.no_grad():

        seq_num = 0

        for x, y in docx_test_loader:

            pred = model(x)
            loss = criterion(pred, y.long()).squeeze()
            soft_max = functional.softmax(pred, 1)
            res = torch.argmax(soft_max, dim=1)
            print('test loss:', loss.item())
            non_zero = count_nonzero(x.numpy() - res.numpy())
            original = cdict.decode(x.squeeze())
            predicted = cdict.decode(res.squeeze())
            if non_zero is not 0:
                # tagger.add_error(docx_test.file_name, seq_num, original, predicted)
                printer.add_line(original, predicted)
            else: 
                printer.add_line(original, -1)
            seq_num += 1

        # tagger.print_tree_to_file()
        printer.print_to_file()
