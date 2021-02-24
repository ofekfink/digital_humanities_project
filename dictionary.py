import torch


# ignores cases where char not in dictionary
class CharDictionary:

    source_file = "FILES/HUGE_TEXTS/merged_xmls.txt"

    def __init__(self):
        with open(self.source_file, "r") as sf:
            self.char_dict = {char: index for index, char in enumerate(sorted(set(sf.read())))}

    def encode(self, sentence):
        return torch.tensor([self.char_dict[char] for char in sentence], dtype=torch.long)

    def decode(self, tensor):
        index_to_char = {index: char for char, index in self.char_dict.items()}
        return "".join([index_to_char[int(t)] for t in tensor])


