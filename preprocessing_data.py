import os
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, PorterStemmer, pos_tag
from nltk_preprocess import make_dictionary, vectorization


def preprocessing_nltk(txt_data):
    # Lowercase
    lower_data = txt_data.lower()

    # removing punctuation
    # print("all punctuations: {}".format(string.punctuation))
    punctuation_data = "“”‘’…{}".format(string.punctuation)

    text_p = "".join([char for char in lower_data if char not in punctuation_data])

    # tokenize
    words = word_tokenize(text_p)

    # stopwords filtering
    stop_words = stopwords.words('english')
    # print("stopwords: {}".format(stopwords))
    filtered_words = [word for word in words if word not in stop_words]
    # print(filtered_words)
    # ['truth', 'universally', 'acknowledged', 'single', 'man', 'possession', 'good', 'fortune', 'must', 'want', 'wife']

    # Stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_words]
    # print(stemmed)
    # ['truth', 'univers', 'acknowledg', 'singl', 'man', 'possess', 'good', 'fortun', 'must', 'want', 'wife']

    # pos tag
    pos = pos_tag(filtered_words)
    # print(pos)
    # [('truth', 'NN'), ('universally', 'RB'), ('acknowledged', 'VBD'), ('single', 'JJ'), ('man', 'NN'),
    # ('possession', 'NN'), ('good', 'JJ'), ('fortune', 'NN'), ('must', 'MD'), ('want', 'VB'), ('wife', 'NN')]
    return words, filtered_words, stemmed, pos


def read_txt(txt_file):
    try:
        with open(txt_file, "rt", encoding="utf-8") as r_f:
            return [line.strip() for line in r_f.readlines()]
    except:
        with open(txt_file, "rt") as r_f:
            return [line.strip() for line in r_f.readlines()]


def read_phase(txt_file):
    all_lines = read_txt(txt_file)
    all_phases = []
    cur_phase = []
    for line in all_lines:
        if line:
            cur_phase.append(line)
        else:
            if cur_phase:
                all_phases.append(" ".join(cur_phase))
            cur_phase = []

    if cur_phase:
        all_phases.append(" ".join(cur_phase))

    return all_phases


def read_all_txt(txt_file):
    all_phases = read_phase(txt_file)
    return " ".join(all_phases)


def processing_txt(txt_dir):
    all_txt = []
    all_file_names = []
    for txt_file in os.listdir(txt_dir):
        if txt_file.find(".txt") == -1:
            continue

        # if txt_file != "g1pB_taska.txt":
        #    continue

        print("txt file: {}".format(txt_file))
        txt_data = read_all_txt(os.path.join(txt_dir, txt_file))
        _, filtered_words, _, _ = preprocessing_nltk(txt_data)
        all_txt.append(filtered_words)
        all_file_names.append(txt_file)

    # make test data
    dict_data = make_dictionary(all_txt)
    for text, txt_file in zip(all_txt, all_file_names):
        text_vect = vectorization(text, dict_data)
        print("txt file: {}".format(txt_file))
        print("text: {}".format(text))
        print("text vector: {}".format(text_vect))


if __name__ == "__main__":
    text_dir = "data"
    processing_txt(text_dir)