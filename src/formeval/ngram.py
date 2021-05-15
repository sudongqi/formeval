import collections


def get_ngram_counts(data, n):
    res = [collections.defaultdict(list) for _ in range(n)]
    for key, sentences in data.items():
        for sentence_in_tokens in sentences:
            for i in range(n):
                count = ngram_counts(sentence_in_tokens, i + 1)
                res[i][key].append(count)
    return res


def ngram_counts(tokens, n):
    if n == 1:
        return collections.Counter(tokens)
    return collections.Counter(' '.join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)) \
        if len(tokens) >= n else {}
