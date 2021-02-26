from random import randint, sample
from dictionary import char_dict
import numpy as np


# a class to create intentional errors in correct xmls
class ErrorsMaker:
    swaps = dict.fromkeys(char_dict.keys(), [])  # char -> similar char list
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
        ['!', 'ן'],
    ]
    error_rate = 0.2

    def __init__(self):
        self.build_swaps_dict()
        self.errors_factory = {
            0: self.swap_adjacent_letters,
            1: self.swap_similar_letters,
            2: self.remove_needed_space
        }

    # builds a table that stores all possible swaps per letter
    def build_swaps_dict(self):
        for pair in self.similar_letter:
            self.swaps[pair[0]].append(pair[1])
            self.swaps[pair[1]].append(pair[0])

    # =========================================
    def get_indices(self, text_len):
        indexes = np.arange(text_len)
        amount = int(text_len * self.error_rate)
        indices = sample(indexes, amount)
        return indices

    def fault_text(self, text):
        indices = self.get_indices(len(text))
        new_text = ""
        for i in indices:
            if text[i] is ' ':
                pass  # remove the space from text as an error,
            else:



    # =========================================

    # todo check if pass by val
    def swap_similar_letters(self, word):
        index = randint(0, len(word) - 1)
        letter_to_swap = word[index]
        try:
            similars = self.swaps[letter_to_swap]
        except KeyError:
            print("word {} letter {} should have been swapped".format(word, letter_to_swap))
            return word
        swap_to = choice(similars)
        lst = list(word)
        lst[index] = swap_to
        return ''.join(lst)

    @staticmethod
    def swap_adjacent_letters(word):
        index = 0 if len(word) < 3 else randint(0, len(word) - 2)
        lst = list(word)
        lst[index], lst[index + 1] = lst[index + 1], lst[index]
        return ''.join(lst)

    # todo
    @staticmethod
    def remove_needed_space(word):
        return word

    def dispatch_error(self, name):
        return self.errors_factory[name]
