def new_error_line(line, predicted):
    line = line.split(" ")
    predicted = predicted.split(" ")
    word_indexes = [i for i in range(len(line)) if line[i] != predicted[i]]
    for i in word_indexes:
        line[i] = '<error>' + line[i] + '</error>'
    return ' '.join(line)


class Printer:

    def __init__(self):
        self.to_print = ''
        self.file_name = 'output.txt'

    def add_line(self, line, predicted):
        if predicted == -1:
            self.to_print += line + '\n'
        else:
            self.to_print += new_error_line(line, predicted) + '\n'

    def print_to_file(self):
        with open(self.file_name, "w+", encoding='UTF-8') as f:
            f.write(self.to_print)
            f.close()
