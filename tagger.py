from lxml import etree


class Tagger:

    def __init__(self):
        self.file_name = 'output.xml'
        file = open(self.file_name, 'w+', encoding="UTF-8")
        tree = etree.ElementTree(etree.Element("errors"))
        tree.write(self.file_name, xml_declaration=True, encoding='UTF-8')
        file.close()
        self.root = etree.parse(self.file_name).getroot()

    def add_error(self, rule_file_name, line_number, error, prediction):
        error = create_error(rule_file_name, line_number, error, prediction)
        self.root.append(error)

    def print_tree_to_file(self):
        tree = etree.tostring(self.root, encoding='UTF-8', xml_declaration=True, pretty_print=True)
        with open(self.file_name, "wb") as f:
            f.write(tree)
            f.close()


def create_error(rule_file_name, line_number, error_line, predicted_line):
    differences = get_difference_word(error_line, predicted_line)

    error = etree.Element("error")
    etree.SubElement(error, "original").text = error_line
    etree.SubElement(error, "predicted").text = predicted_line
    diffs = etree.SubElement(error, "differences")
    for diff in differences:
        etree.SubElement(diffs, "difference").text = diff

    error.set('file_name', rule_file_name)
    error.set('line', str(line_number))

    return error


def get_difference_word(str1, str2):
    array1 = str1.split(" ")
    array2 = str2.split(" ")
    differences = ['in word {}, was "{}", and predicted "{}"'.format(i, array1[i], array2[i]) for i in range(len(array1)) if array1[i] != array2[i]]
    return differences
