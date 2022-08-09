from .utils import *
from .utils import _mean, _harmonic_mean, _max


class BaseEvaluator:

    def __init__(self, references, processor, already_processed, silent=True, has_instance_score=True):
        self.has_instance_score = has_instance_score
        self.references = references
        self.processor = processor
        self.already_processed = already_processed
        self.silent = silent
        self.timer = Timer()
        self.log_data_stats(self.references, 'references')

    def evaluate(self, candidates):
        raise NotImplementedError

    def _process(self, data):
        self.timer.check()
        res = self.processor.process(data) if not self.already_processed else data
        self._log('processor.process(data) took {:3f}s'.format(self.timer.check()))
        return res

    def get_name(self, detailed=False):
        raise NotImplementedError

    def _log(self, message):
        if not self.silent:
            log(message)

    def log_data_stats(self, data, name='data'):
        self._log('<{}> has {} unique ids and {} sentences'.format(name, len(data), sum(map(len, data.values()))))

    def log_setup_finish(self):
        self._log('<references> processing completed, took {:.3f}s'.format(self.timer.total_time()))

    def log_evaluation_start(self, candidates):
        if not set(candidates.keys()).issubset(set(self.references.keys())):
            raise UnknownCandidatesError
        self.timer = Timer()
        self.log_data_stats(candidates, 'candidates')

    def log_evaluation_finish(self):
        self._log('<candidates> evaluation completed, took {:.3f}s'.format(self.timer.total_time()))

    def aggregate_scores(self, scores):
        return _mean([s for _scores in scores.values() for s in _scores])

    def references_agreement(self, upper_bound=False, num_trials=10, discard_identical=True,
                             self_agreement_n_samples=None, **kwargs):
        instance_scores = collections.defaultdict(list)
        scores = []
        for i in range(num_trials):
            candidates, references = split_references(self.references, discard_identical=discard_identical)
            if self_agreement_n_samples is not None:
                references = sample_references(references, self_agreement_n_samples)
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
                       reference_scores_upper_bound=None,
                       max_num_references=None,
                       self_agreement_n_samples=None,
                       ):
        candidate_score, candidate_scores = self.evaluate(candidates)

        if reference_scores is None or reference_score is None:
            reference_score, reference_scores = self.references_agreement(
                self_agreement_n_samples=self_agreement_n_samples)
        if reference_scores_upper_bound is None or reference_score_upper_bound is None:
            reference_score_upper_bound, reference_scores_upper_bound = self.references_agreement(
                upper_bound=True, self_agreement_n_samples=self_agreement_n_samples)

        column_names = ['aggregation method', 'score', 'self-agreement', 'self-agreement (upper bound)']
        rows = [
            ['default', candidate_score, reference_score, reference_score_upper_bound],
            ['mean', _mean(candidate_scores), _mean(reference_scores), _mean(reference_scores_upper_bound)],
            ['harmonic mean', _harmonic_mean(candidate_scores), _harmonic_mean(reference_scores),
             _harmonic_mean(reference_scores_upper_bound)]
        ]

        res = build_table(rows, column_names=column_names)
        log('\n'.join(res))

        with logger(file=open(path, 'w')):
            log(self.get_name(detailed=True) + '\n')
            sep(90)
            print_iter(res)
            sep(90)
            for key, _candidates in candidates.items():
                for candidate, can_score, ref_score in zip(_candidates, candidate_scores[key], reference_scores[key]):
                    log('id: {}\n\n'.format(key))
                    log('candidate:\n----> {}\n'.format(candidate))
                    log('references:\n{}\n\n'.format('\n'.join(['===== ' + s for s in self.references[key]])))
                    log('candidate score: {:.3f}\n'.format(can_score))
                    log('reference score: {:.3f}\n'.format(ref_score))
                    sep(30)
