PUNCTUATIONS = {"''", "'", "``", "`", ".", "?", "!", ",", ":", "-", "--", "...", ";"}


class RegexWordnetProcessor:
    def __init__(self):
        from .utils import get_regex_tokenizer, get_wordnet_lemmatizer
        self.tokenizer = get_regex_tokenizer()
        self.lemmatizer = get_wordnet_lemmatizer()

    def process(self, data):
        res = {}
        for key, sentences in data.items():
            res[key] = [[self.lemmatizer.lemmatize(token.lower())
                         for token in self.tokenizer.tokenize(s) if token not in PUNCTUATIONS]
                        for s in sentences]
        return res


class RegexProcessor:
    def __init__(self):
        from .utils import get_regex_tokenizer
        self.tokenizer = get_regex_tokenizer()

    def process(self, data):
        res = {}
        for key, sentences in data.items():
            res[key] = [[token.lower() for token in self.tokenizer.tokenize(s) if token not in PUNCTUATIONS]
                        for s in sentences]
        return res


class NaiveProcessor:
    def __init__(self):
        pass

    def process(self, data):
        res = {}
        for key, sentences in data.items():
            res[key] = [[token.lower() for token in s.split() if token not in PUNCTUATIONS] for s in sentences]
        return res
