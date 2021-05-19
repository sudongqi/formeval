import collections

from formeval.utils import SimpleTimer, UnknownCandidatesError, _mean, _max, sample_from_references, _harmonic_mean


class BaseEvaluator:

    def __init__(self, references, processor, already_processed, silent, has_instance_score=True):
        self.has_instance_score = has_instance_score
        self.references = references
        self.processor = processor
        self.already_processed = already_processed
        self.silent = silent
        self.timer = SimpleTimer()
        self.log_data_stats(self.references, 'references')

    def evaluate(self, candidates):
        raise NotImplementedError

    def _process(self, data):
        self.timer.check()
        res = self.processor.process(data) if not self.already_processed else data
        self.log('processor.process(data) took {:3f}s'.format(self.timer.check()))
        return res

    def get_name(self, detailed=False):
        raise NotImplementedError

    def log(self, message):
        if not self.silent:
            print(message)

    def log_data_stats(self, data, name='data'):
        self.log('{} has {} unique ids and {} sentences'.format(name, len(data), sum(map(len, data.values()))))

    def log_setup_finish(self):
        self.log('references processing completed, took {:.3f}s'.format(self.timer.total_time()))

    def log_evaluation_start(self, candidates):
        if not set(candidates.keys()).issubset(set(self.references.keys())):
            raise UnknownCandidatesError
        self.timer = SimpleTimer()
        self.log_data_stats(candidates, 'candidates')

    def log_evaluation_finish(self):
        self.log('candidates evaluation completed, took {:.3f}s'.format(self.timer.total_time()))

    def aggregate_scores(self, scores):
        return _mean([s for _scores in scores.values() for s in _scores])

    def references_agreement(self, upper_bound=False):
        return self.references_agreement_by_sampling(upper_bound)

    def references_agreement_by_sampling(self, upper_bound=False, num_trials=10, discard_identical=True, **kwargs):
        instance_scores = collections.defaultdict(list)
        scores = []
        for i in range(num_trials):
            candidates, references = sample_from_references(self.references, discard_identical=discard_identical)
            evaluator = self.__class__(references=references, **kwargs)
            score, _scores = evaluator.evaluate(candidates)
            scores.append(score)
            for key, score_list in _scores.items():
                instance_scores[key].extend(score_list)
        f = _max if upper_bound else _mean
        instance_scores = {key: [f(value)] for key, value in instance_scores.items()}
        _score = self.aggregate_scores(instance_scores) if self.has_instance_score else _mean(scores)
        return _score, instance_scores

    def compile_report(self, path, candidates, reference_score=None, reference_scores=None,
                       reference_score_upper_bound=None,
                       reference_scores_upper_bound=None
                       ):

        candidate_score, candidate_scores = self.evaluate(candidates)
        if reference_scores is None or reference_score is None:
            reference_score, reference_scores = self.references_agreement()
        if reference_scores_upper_bound is None or reference_score_upper_bound is None:
            reference_score_upper_bound, reference_scores_upper_bound = self.references_agreement(upper_bound=True)

        res = ['aggregate method: candidate | references | upper bound\n',
               'mean            : {:.3f} | {:.3f} | {:.3f}\n'.format(_mean(candidate_scores), _mean(reference_scores),
                                                                     _mean(reference_scores_upper_bound)),
               'harmonic mean   : {:.3f} | {:.3f} | {:.3f}\n'.format(_harmonic_mean(candidate_scores),
                                                                     _harmonic_mean(reference_scores),
                                                                     _harmonic_mean(reference_scores_upper_bound)),
               'default         : {:.3f} | {:.3f} | {:.3f}\n'.format(candidate_score, reference_score,
                                                                     reference_score_upper_bound)
               ]

        with open(path, 'w') as f:
            f.write(self.get_name(detailed=True) + '\n')
            f.write('-----------------------------------------------\n')
            for r in res:
                f.write(r)
            f.write('-----------------------------------------------\n')
            for key, _candidates in candidates.items():
                for candidate, can_score, ref_score in zip(_candidates, candidate_scores[key], reference_scores[key]):
                    f.write('id: {}\n\n'.format(key))
                    f.write('candidate:\n----> {}\n'.format(candidate))
                    f.write('references:\n{}\n\n'.format('\n'.join(['===== ' + s for s in self.references[key]])))
                    f.write('candidate score: {:.3f}\n'.format(can_score))
                    f.write('reference score: {:.3f}\n'.format(ref_score))
                    f.write('-----------------------------------------------\n')
        return res
