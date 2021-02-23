import zipfile
import xml.etree.ElementTree as ET
import docx2txt
import os
import re


class Text:

    @staticmethod
    def preprocess(text):
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = re.sub("\n +", '\n', text)
        text = re.sub("\n+", '\n', text)
        return text


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
        text = self.preprocess(text)
        return text


class Docxs(Text):

    docxs = ["FILES/DOCX/" + f for f in os.listdir("FILES/DOCX/")]

    def get_docx_text(self, docx_file):
        text = docx2txt.process(docx_file)
        text = self.preprocess(text)
        return text
