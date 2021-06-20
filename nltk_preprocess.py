from collections import defaultdict
from gensim import corpora, similarities


def creating_corpus():

    documents = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]

    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]


def make_dictionary(texts):
    dictionary = corpora.Dictionary(texts)
    feature_len = len(dictionary.token2id)
    return dictionary, feature_len


def vectorization(text, dictionary):
    return dictionary.doc2bow(text)


def tfidf_transform(corpus):
    from gensim import models

    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
    return tfidf


def vector_by_tfidf(bow_vect, tfidf):
    tfidf_vect = tfidf[bow_vect]
    return tfidf_vect


def text_similarity(tfidf_vect1, tfidf_vect2, feature_len):
    index = similarities.SparseMatrixSimilarity([tfidf_vect1], num_features=feature_len)
    sim = index[tfidf_vect2]
    return sim


def transforming_tfidf_vecot(corpus, tfidf):
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)


if __name__ == "__main__":
    creating_corpus()
