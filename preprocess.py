import zipfile
import xml.etree.ElementTree as ET
import docx2txt
import os
import re
from abc import ABC, abstractmethod


class Text(ABC):

    # removes double spaces and newlines
    @staticmethod
    def preprocess(text):
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = re.sub("\n +", '\n', text)
        text = re.sub("\n+", '\n', text)
        return text

    # converts all files in list to text after preprocess
    def to_text_files(self, files):
        count = 0
        for file in files:
            new_file = self.new_file_name(file)
            text = self.get_text(file)
            try:
                text_file = open(new_file, "w+")
                text_file.write(text)
            except (UnicodeEncodeError, FileNotFoundError):
                count = count + 1
                continue
            text_file.close()
        print("could not save : {} files".format(count))

    @abstractmethod
    def get_text(self, file):
        pass

    @abstractmethod
    def new_file_name(self, file):
        pass

    # converts all texts in dir to one big text file
    @staticmethod
    def to_one_file(directory, output_path):
        files = [directory + "/" + f for f in os.listdir(directory)]
        with open(output_path, 'w') as outfile:
            for file in files:
                with open(file) as infile:
                    for line in infile:
                        outfile.write(line)


# could not save 114 xmls
class Xmls(Text):
    akn = "FILES/DRIVE/LawRepoWiki_17_1_20.zip"
    zip_akn = zipfile.ZipFile(akn, 'r')
    xmls = [item for item in zip_akn.namelist() if ".xml" in item and "law_dictionary.xml" not in item]

    def get_text(self, xml_file):
        f = self.zip_akn.open(xml_file)
        tree = ET.parse(f)
        root = tree.getroot()
        iter = list(root.iter())
        text = ' '.join([element.text or "" for element in iter])
        text = self.preprocess(text)
        return text

    # generates a text file name for that xml text
    @staticmethod
    def new_file_name(xml):
        slashes = xml.split("/")
        new_file_name = "FILES/XML_TO_TEXT/" + "_".join(slashes[3:6]) + ".txt"
        return new_file_name


# could not save 0 xmls
class Docxs(Text):
    docxs = ["FILES/DOCX/" + f for f in os.listdir("FILES/DOCX/")]

    @staticmethod
    def new_file_name(docx):
        name = docx.split(".")[0].split("/")[2]
        new_file_name = "FILES/DOCX_TO_TEXT/" + name + ".txt"
        return new_file_name

    # generates a text file name for that docx text
    def get_text(self, docx_file):
        text = docx2txt.process(docx_file)
        text = self.preprocess(text)
        return text
