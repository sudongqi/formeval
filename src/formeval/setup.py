import nltk


def setup_everything():
    setup_wordnet()
    setup_spice()


def setup_wordnet():
    nltk.download('wordnet')


def setup_spice():
    pass
