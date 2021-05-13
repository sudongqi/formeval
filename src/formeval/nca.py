import collections
from .evaluator import FormEvaluator
from .processor import FormProcessor, lemminfect_lemmatizer, COMMON_DETERMINERS
from .utils import harmonic_mean, mean


def get_common_concepts(sets, concept_threshold, concept_threshold_ratio):
    counter = collections.Counter()
    set_size = len(sets)
    _concept_threshold = min(set_size, max(concept_threshold, int(concept_threshold_ratio * set_size)))
    for _set in sets:
        for t in set(_set):
            counter[t] += 1
    return set(k for k, v in counter.items() if v >= _concept_threshold) - COMMON_DETERMINERS


def summary(candidate, references):
    can = candidate if isinstance(candidate, str) else ' '.join(candidate)
    refs = references if isinstance(references[0], str) else [' '.join(r) for r in references]
    refs = ['===== ' + r for r in refs]
    return 'pred:\n----> {}\nreferences:\n{}\n'.format(can, '\n'.join(refs))


def clean(sentence, concepts):
    return [w for w in sentence if w in concepts]


def get_counter_max_item(counter):
    return sorted([(v, k) for k, v in counter.items()])[-1][1]


def collect_nca(concepts, empty_token='<empty>'):
    if len(concepts) == 0 or sum(len(c) for c in concepts) == 0:
        return set()
    concepts = [concepts] if isinstance(concepts[0], str) else concepts
    counter = collections.defaultdict(collections.Counter)
    for _concepts in concepts:
        for i in range(len(_concepts)):
            c = _concepts[i]
            left_c = empty_token if i == 0 else _concepts[i - 1]
            right_c = empty_token if i == len(_concepts) - 1 else _concepts[i + 1]
            counter[c + '_l'][left_c] += 1
            counter[c + '_r'][right_c] += 1
    return set(k + '_' + get_counter_max_item(v) for k, v in counter.items())


def nca_f1(candidate, reference):
    precision = sum([1 for p in candidate if p in reference]) / len(candidate)
    recall = sum([1 for p in reference if p in candidate]) / len(reference)
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def references_agreement_nca_score(references, concept_threshold=4, concept_threshold_ratio=0.5):
    scores = []
    for i in range(len(references)):
        _can = references[i]
        _ref = references[:i] + references[i + 1:]
        scores.append(get_concepts_and_evaluate(_can, _ref, concept_threshold, concept_threshold_ratio))
    return mean(scores)


def candidate_nca_score(candidate, references, concept_threshold=4, concept_threshold_ratio=0.5):
    scores = []
    for i in range(len(references)):
        _references = references[:i] + references[i + 1:]
        scores.append(get_concepts_and_evaluate(candidate, _references, concept_threshold, concept_threshold_ratio))
    return mean(scores)


def get_concepts_and_evaluate(candidate, references, concept_threshold, concept_threshold_ratio):
    concepts = get_common_concepts(references, concept_threshold, concept_threshold_ratio)
    candidate_concept = clean(candidate, concepts)
    references_concept = [clean(r, concepts) for r in references]
    candidate_nca = collect_nca(candidate_concept)
    references_nca = collect_nca(references_concept)
    if len(candidate_nca) == 0:
        return 0
    elif len(references_nca) == 0:
        return 1
    return nca_f1(candidate_nca, references_nca)


class NCAEvaluator(FormEvaluator):

    def __init__(self, references, processor=None, already_processed=False, silent=True,
                 concept_threshold=4,
                 concept_threshold_ratio=0.5,
                 harmonic_mean_min_value=0.01):
        super(NCAEvaluator, self).__init__(references=references,
                                           processor=processor if processor else FormProcessor(
                                               lemmatizer=lemminfect_lemmatizer()),
                                           already_processed=already_processed,
                                           silent=silent
                                           )
        self.references = references
        self._references = self._process(references)
        self.concept_threshold = concept_threshold
        self.concept_threshold_ratio = concept_threshold_ratio
        self.harmonic_mean_min_value = harmonic_mean_min_value
        self.log_build_references_timer()

    def evaluate(self, candidates):
        self.check_candidates_and_start_evaluation_timer(candidates)
        scores = {}
        for key, _candidates in self._process(candidates).items():
            scores[key] = [candidate_nca_score(candidate,
                                               self._references[key],
                                               self.concept_threshold,
                                               self.concept_threshold_ratio)
                           for candidate in _candidates]
        self.log_evaluation_timer()
        return self.aggregate_scores(scores), scores

    def aggregate_scores(self, scores):
        return harmonic_mean([s for _scores in scores.values() for s in _scores], self.harmonic_mean_min_value)

    def references_agreement(self):
        scores = {key: [references_agreement_nca_score(references,
                                                       self.concept_threshold,
                                                       self.concept_threshold_ratio)]
                  for key, references in self._references.items()}
        return self.aggregate_scores(scores), scores

    def get_name(self, detailed=True):
        res = 'nca'
        if detailed:
            res += ' (concept_threshold={}, concept_threshold_ratio={}, harmonic_mean_min_value={})'.format(
                self.concept_threshold,
                self.concept_threshold_ratio,
                self.harmonic_mean_min_value)
        return res
