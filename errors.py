from random import randint, sample, choice
from dictionary import cdict
import numpy as np


# a class to create intentional errors in correct xmls
class ErrorsMaker:

    swaps = dict.fromkeys(cdict.char_dict.keys(), [])  # char -> similar char list
    similar_letter = [
        ['מ', 'נ'],
        ['ם', 'מ'],
        ['ן', 'ך'],
        ['ט', 'פ'],
        ['פ', 'ס'],
        ['ך', 'ד'],
        ['ג', 'נ'],
        ["'", 'י'],
        ['?', 'ל'],
        ['ט', 'ם'],
        ['ב', 'נ'],
        # ['!', 'ן'],
    ]
    error_rate = 0.2

    def __init__(self):
        self.build_swaps_dict()
        self.errors_factory = {
            0: self.swap_adjacent_letters,
            1: self.swap_similar_letters,
            # 2: self.remove_needed_space
        }

    # builds a table that stores all possible swaps per letter
    def build_swaps_dict(self):
        for pair in self.similar_letter:
            self.swaps[pair[0]].append(pair[1])
            self.swaps[pair[1]].append(pair[0])

    def get_indices(self, text_len):
        amount = int(text_len * self.error_rate)
        indices = sample(range(text_len-1), amount)
        indices = sorted(indices)
        return indices

    def dispatch_error(self, error_num):
        return self.errors_factory[error_num]

    def swap_similar_letters(self, char_list, index):
        letter_to_swap = char_list[index]
        try:
            similars = self.swaps[letter_to_swap]
        except KeyError:
            print("char_list {} letter {} should have been swapped".format(char_list, letter_to_swap))
            return char_list
        swap_to = choice(similars)
        char_list[index] = swap_to
        return char_list

    @staticmethod
    def swap_adjacent_letters(char_list, index):
        char_list[index], char_list[index + 1] = char_list[index + 1], char_list[index]  # TODO check
        return char_list

    def fault_text(self, text):
        indices = self.get_indices(len(text))
        labels = [char for char in text]
        faulted = [char for char in text]
        for i in indices:
            char = text[i]
            if char == ' ' or char == '\n':
                pass
            else:
                error_type = randint(0, 1)
                error_func = self.dispatch_error(error_type)
                faulted = error_func(faulted, i)
        faulted = "".join(faulted)
        labels = "".join(labels)
        return faulted, labels

