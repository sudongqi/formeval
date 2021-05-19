import nltk
from lemminflect import Lemmatizer
from functools import lru_cache

PUNCTUATIONS = {"''", "'", "``", "`", ".", "?", "!", ",", ":", "-", "--", "...", ";"}
COMMON_DETERMINERS = {'a', 'an', 'the'}


class _WordNetLemmatizer:
    def __init__(self):
        self.model = nltk.wordnet.WordNetLemmatizer()

    def lemmatize(self, word, pos):
        return self.model.lemmatize(word, 'v') if pos is None else self.model.lemmatize(word, pos)


def wordnet_lemmatizer():
    return _WordNetLemmatizer()


class _SnowballStemmer:
    def __init__(self, language):
        self.model = nltk.stem.snowball.SnowballStemmer(language)

    def lemmatize(self, word):
        return self.model.stem(word)


def snowball_lemmatizer(language='english'):
    return _SnowballStemmer(language)


class _PorterStemmer:
    def __init__(self):
        self.model = nltk.stem.porter.PorterStemmer()

    def lemmatize(self, word):
        return self.model.stem(word)


class _LemmInfectLemmatizer:
    def __init__(self):
        self.model = Lemmatizer()

    @lru_cache(maxsize=100000)
    def lemmatize(self, word, upos=None):
        return self.model.getLemma(word, 'VERB')[0] if upos is None else self.model.getLemma(word, upos)[0]


def lemminfect_lemmatizer():
    return _LemmInfectLemmatizer()


def porter_lemmatizer():
    return _PorterStemmer()


def regexp_tokenizer():
    return nltk.tokenize.regexp.RegexpTokenizer('\w+|\$[\d\.]+|\S+')


class FormProcessor:
    def __init__(self, tokenizer=None, lemmatizer=None):
        self.tokenizer = tokenizer if tokenizer is not None else regexp_tokenizer()
        self.lemmatizer = lemmatizer

    def _process(self, sentence):
        return [w.lower() if self.lemmatizer is None else self.lemmatizer.lemmatize(w.lower())
                for w in self.tokenizer.tokenize(sentence) if w not in PUNCTUATIONS]

    def process(self, data):
        if isinstance(data, str):
            return self._process(data)
        elif isinstance(data, list):
            return [self._process(d) for d in data]
        elif isinstance(data, dict):
            return {key: [self._process(s) for s in sentences] for key, sentences in data.items()}


def get_sets_intersection(sets):
    res = set(sets[0]) - COMMON_DETERMINERS
    for i in range(1, len(sets)):
        res = res.intersection(sets[i])
    return res
