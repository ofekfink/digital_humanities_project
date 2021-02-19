import zipfile
import xml.etree.ElementTree as ET
import string
from random import sample, randint, choice
import docx2txt
import os


# a class to create intentional errors in correct xmls
class Errors:

    errors_makers = {}
    hebrew_letter = [chr(c) for c in range(0X5D0, 0X5EB)] + ['?', '!', ',', "'", "\"", "”"]
    swaps = dict.fromkeys(hebrew_letter, [])
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

    def __init__(self):
        self.build_swaps_dict()
        self.errors_makers = {
            0: self.swap_adjacent_letters,
            1: self.swap_similar_letters,
            2: self.remove_needed_space
        }

    # builds a table that stores all possible swaps per letter
    def build_swaps_dict(self):
        for pair in self.similar_letter:
            self.swaps[pair[0]].append(pair[1])
            self.swaps[pair[1]].append(pair[0])

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
        return self.errors_makers[name]


# a class to open, read and deal with texts from docxs and xmls
class Text:

    chars_to_clean = string.punctuation + string.digits
    hebrew_letter = [chr(c) for c in range(0X5D0, 0X5EB)]
    errors = Errors()
    error_rate = 30

    # text -> text, clean special chars and whitespaces
    def preprocess_text(self, text):
        # remove all non hebrew characters (except \n and space)
        text = text.translate(str.maketrans('', '', self.chars_to_clean))
        text = text.replace("–", " ").replace("-", " ").replace("־", " ")

        # clean start & end whitespaces, remove empty lines and split to sentences
        sentences = [s.strip() for s in text.splitlines() if s and not s.isspace()]

        # remove sentences of one word one letter (usually for subsections in laws)
        sentences = [s for s in sentences if s if not (len(s) == 1 and len(s[0]) == 1)]

        # clean one letter words
        clean_sentences = []
        for sentence in sentences:
            words = sentence.split()
            words = [word for word in words if len(word) > 1]
            sentence = " ".join(word for word in words)
            clean_sentences.append(sentence)
        text = "\n".join(sentence for sentence in clean_sentences)
        return text

    # text -> text, each changed char gets a label 1
    def fault_text(self, text):
        text_as_words = text.split()
        num_words = len(text_as_words)
        num_errors = len(self.errors.errors_makers)
        num_to_fault = int(self.error_rate * num_words / 100)
        errors_indexes = sample(range(num_words), num_to_fault)
        for index in errors_indexes:
            error_kind = randint(0, num_errors - 1)
            faulted_word = self.errors.dispatch_error(error_kind)(text_as_words[index])
            text_as_words[index] = faulted_word
        faulted_text = " ".join(word for word in text_as_words)
        return faulted_text

    @staticmethod
    def flatten(nested_list):
        return [word for sublist in nested_list for word in sublist]


class Xmls(Text):

    akn = "FILES/DRIVE/LawRepoWiki_17_1_20.zip"
    zip_akn = zipfile.ZipFile(akn, 'r')
    xmls = [item for item in zip_akn.namelist() if ".xml" in item and "law_dictionary.xml" not in item]

    def get_xml_text(self, xml_file):
        f = self.zip_akn.open(xml_file)
        tree = ET.parse(f)
        root = tree.getroot()
        iter = list(root.iter())
        text = ' '.join([element.text or "" for element in iter])
        return text

    # returns a list of preprocessed texts
    def get_xmls_texts(self):
        xmls = [item for item in self.zip_akn.namelist() if ".xml" in item]
        xmls_texts = [self.get_xml_text(xml) for xml in xmls]
        xmls_texts = [self.preprocess_text(xml) for xml in xmls_texts]
        return xmls_texts

    def xmls_to_text_files(self):
        count = 0
        for xml in self.xmls:
            slashes = xml.split("/")
            new_file = "XML_TO_TEXT/" + "_".join(slashes[3:6]) + ".txt"
            text = self.get_xml_text(xml)
            clean_text = self.preprocess_text(text)
            try:
                text_file = open(new_file, "w+")
                text_file.write(clean_text)
            except (UnicodeEncodeError, FileNotFoundError):
                count = count + 1
                continue
            text_file.close()
        print("could not save : {} files".format(count))

    def get_xmls_words(self):
        xmls = [item for item in self.zip_akn.namelist() if ".xml" in item]
        xmls_texts = [self.get_xml_text(xml) for xml in xmls]
        xmls_texts = [self.preprocess_text(text) for text in xmls_texts]
        xmls_sentences = [text.splitlines() for text in xmls_texts]
        xmls_sentences = self.flatten(xmls_sentences)
        xml_words = [s.split() for s in xmls_sentences]
        xml_words = self.flatten(xml_words)
        xml_words = [w for w in xml_words if len(w) > 1]
        return xml_words


class Docxs(Text):

    docxs = ["FILES/DOCX/" + f for f in os.listdir("FILES/DOCX/")]

    @staticmethod
    def get_docx_text(docx_file):
        return docx2txt.process(docx_file)

    # returns a list of texts
    def get_docxs_texts(self):
        docxs = [item for item in self.docxs]
        docxs_texts = [self.get_docx_text(docx) for docx in docxs]
        docxs_texts = [self.preprocess_text(xml) for xml in docxs_texts]
        return docxs_texts

    def docxs_to_text_files(self):
        count = 0
        for docx in self.docxs:
            name = docx.split(".")[0].split("/")[1]
            new_file = "DOCX_TO_TEXT/" + name + ".txt"
            text = self.get_docx_text(docx)
            clean_text = self.preprocess_text(text)
            try:
                text_file = open(new_file, "w+")
                text_file.write(clean_text)
            except (UnicodeEncodeError, FileNotFoundError):
                count = count + 1
                continue
            text_file.close()
        print("could not save : {} files".format(count))

    def get_docxs_words(self):
        docxs = [item for item in self.docxs]
        docxs_texts = [self.get_docx_text(docx) for docx in docxs]
        docxs_texts = [self.preprocess_text(docx) for docx in docxs_texts]
        docxs_sentences = [text.splitlines() for text in docxs_texts]
        docxs_sentences = self.flatten(docxs_sentences)
        docxs_words = [s.split() for s in docxs_sentences]
        docxs_words = self.flatten(docxs_words)
        docxs_words = [w for w in docxs_words if len(w) > 1]
        return docxs_words
