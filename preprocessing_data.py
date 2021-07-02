import os
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, PorterStemmer, pos_tag
from nltk_preprocess import make_dictionary, vectorization, tfidf_transform, vector_by_tfidf, text_similarity
from nltk.tokenize import sent_tokenize


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


def all_sents(txt_data):
    all_sent = sent_tokenize(txt_data.lower())
    sent_data = []
    orig_sent = []
    for sent in all_sent:
        words, sent_words, stemmed, pos = preprocessing_nltk(sent)
        sent_data.append(sent_words)
        orig_sent.append(words)

    return orig_sent, sent_data


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


def test_similarity(txt_idx1, txt_idx2, tfidf_model, bow_corpus, feature_len):
    bow_vect1 = bow_corpus[txt_idx1]
    bow_vect2 = bow_corpus[txt_idx2]
    tfidf_vect1 = vector_by_tfidf(bow_vect1, tfidf_model)
    tfidf_vect2 = vector_by_tfidf(bow_vect2, tfidf_model)
    similarity = text_similarity(tfidf_vect1, tfidf_vect2, feature_len)
    return similarity


def test_sent_similarity(txt_idx1, txt_idx2, tfidf_model, sent_corpus, feature_len):
    ref_bow_vects = sent_corpus[txt_idx1]
    comp_bow_vect2 = sent_corpus[txt_idx2]
    cutt_off_prob = 0.3

    # compare similarity
    all_score_pair = []
    for idx_ref, bow_vect1 in enumerate(ref_bow_vects):
        max_similarity = 0
        max_idx = 0
        tfidf_vect1 = vector_by_tfidf(bow_vect1, tfidf_model)
        for idx_doc, bow_vect2 in enumerate(comp_bow_vect2):
            tfidf_vect2 = vector_by_tfidf(bow_vect2, tfidf_model)
            similarity = text_similarity(tfidf_vect1, tfidf_vect2, feature_len)
            if similarity > max_similarity:
                max_similarity = similarity
                max_idx = idx_doc
        if max_similarity > cutt_off_prob:
            all_score_pair.append([max_similarity, max_idx])
    return all_score_pair


def test_all_similarity(all_file_names, all_txt, txt_idx1, tfidf_model, bow_corpus, feature_len):
    for txt_idx in range(len(all_txt)):
        similarity = test_similarity(txt_idx1, txt_idx, tfidf_model, bow_corpus, feature_len)
        print("selected first txt file: {}".format(all_file_names[txt_idx1]))
        print("selected first text: {}".format(all_txt[txt_idx1]))
        print("selected second txt file: {}".format(all_file_names[txt_idx]))
        print("selected second text: {}".format(all_txt[txt_idx]))
        print("similarity of two text: {}".format(similarity))


def test_sent_all_similarity(all_file_names, all_txt, txt_idx1, tfidf_model, sent_bow_corpus, feature_len):
    for txt_idx in range(len(all_txt)):
        all_similarity = test_sent_similarity(txt_idx1, txt_idx, tfidf_model, sent_bow_corpus, feature_len)
        print("selected first txt file: {}".format(all_file_names[txt_idx1]))
        print("selected first text: {}".format(all_txt[txt_idx1]))
        print("selected second txt file: {}".format(all_file_names[txt_idx]))
        print("selected second text: {}".format(all_txt[txt_idx]))
        print("similarity of two text: {}".format(all_similarity))


def processing_txt(txt_dir):
    all_txt = []
    all_file_names = []
    all_sent_words = []
    all_orig_sent_words = []

    print("building corpus")
    for txt_file in os.listdir(txt_dir):
        if txt_file.find(".txt") == -1:
            continue

        # if txt_file != "g1pB_taska.txt":
        #    continue

        # print("txt file: {}".format(txt_file))
        txt_data = read_all_txt(os.path.join(txt_dir, txt_file))
        _, filtered_words, _, _ = preprocessing_nltk(txt_data)
        orig_words, sent_words = all_sents(txt_data)
        all_orig_sent_words.append(orig_words)
        all_sent_words.append(sent_words)
        all_txt.append(filtered_words)
        all_file_names.append(txt_file)

    # make test data
    dict_data, feature_len = make_dictionary(all_txt)
    for text, txt_file in zip(all_txt, all_file_names):
        text_vect = vectorization(text, dict_data)
        #print("txt file: {}".format(txt_file))
        #print("text: {}".format(text))
        #print("text vector: {}".format(text_vect))
    bow_corpus = [vectorization(text, dict_data) for text in all_txt]
    tfidf_model = tfidf_transform(bow_corpus)

    # calculate bow by sentence
    doc_bow_corpus = []
    for doc_txt in all_sent_words:
        sent_bow_corpus = [vectorization(text, dict_data) for text in doc_txt]
        doc_bow_corpus.append(sent_bow_corpus)

    print("all txt files: {}".format(all_file_names))
    print("you can change the value of txt_idx1, txt_idx2 from 0 to {}".format(len(all_file_names)-1))
    ref_idx = 1
    comp_idx = 10
    # sim_val = test_similarity(txt_idx1, txt_idx2, tfidf_model, bow_corpus, feature_len)
    # test_all_similarity(all_file_names, all_txt, txt_idx1, tfidf_model, bow_corpus, feature_len)
    # test_sent_all_similarity(all_file_names, all_txt, txt_idx1, tfidf_model, doc_bow_corpus, feature_len)
    all_similarity = test_sent_similarity(ref_idx, comp_idx, tfidf_model, doc_bow_corpus, feature_len)
    print("selected first txt file: {}".format(all_file_names[ref_idx]))
    print("selected first text: {}".format(all_txt[ref_idx]))
    print("selected second txt file: {}".format(all_file_names[comp_idx]))
    print("selected second text: {}".format(all_txt[comp_idx]))
    print("similarity of two text: {}".format(all_similarity))
    similar_sentence = []
    for score, best_idx in all_similarity:
        similar_sentence.append(" ".join(all_orig_sent_words[comp_idx][best_idx]))

    print("similar sentences: {}".format(similar_sentence))


if __name__ == "__main__":
    text_dir = "data"
    processing_txt(text_dir)
