import torch


class CharDictionary:

    source_files = ["FILES/SMALL_TRAINING/xml_train.txt", "FILES/SMALL_TRAINING/xml_test.txt"] + \
                   ["FILES/DOCX_TO_TEXT/146842.txt"]

    def __init__(self):
        chars = set()
        for file in self.source_files:
            with open(file, 'r') as source_file:
                chars = chars.union(set(source_file.read()))
        chars = sorted(chars)
        self.char_dict = {char: index for index, char in enumerate(chars)}
        # self.cdict = {char: index+1 for index, char in enumerate(chars)}

    def encode(self, sentence):
        encoded = [self.char_dict[char] for char in sentence]
        # encoded = [self.cdict[char] if char in self.cdict else 0 for char in sentence]
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, tensor):
        index_to_char = {index: char for char, index in self.char_dict.items()}
        return "".join([index_to_char[int(t)] for t in tensor])

    def get_vocab_size(self):
        return len(self.char_dict)


cdict = CharDictionary()
