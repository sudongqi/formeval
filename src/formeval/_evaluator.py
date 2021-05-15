import collections

from formeval.utils import SimpleTimer, UnknownCandidatesError, mean, sample_from_references, harmonic_mean


class BaseEvaluator:

    def __init__(self, references, processor, already_processed, silent):
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

    def log_build_references_timer(self):
        self.log('references processing completed, took {:.3f}s'.format(self.timer.total_time()))

    def check_candidates_and_start_evaluation_timer(self, candidates):
        if not set(candidates.keys()).issubset(set(self.references.keys())):
            raise UnknownCandidatesError
        self.timer = SimpleTimer()
        self.log_data_stats(candidates, 'candidates')

    def log_evaluation_timer(self):
        self.log('candidates evaluation completed, took {:.3f}s'.format(self.timer.total_time()))

    def aggregate_scores(self, scores):
        return mean([s for _scores in scores.values() for s in _scores])

    def references_agreement(self):
        return self.references_agreement_by_sampling()

    def references_agreement_by_sampling(self, num_trials=10, discard_identical=True, **kwargs):
        scores = collections.defaultdict(list)
        for i in range(num_trials):
            candidates, references = sample_from_references(self.references, discard_identical=discard_identical)
            evaluator = self.__class__(references=references, **kwargs)
            score, _scores = evaluator.evaluate(candidates)
            for key, score in _scores.items():
                scores[key].extend(score)
        scores = {key: [mean(value)] for key, value in scores.items()}
        return self.aggregate_scores(scores), scores

    def compile_report(self, path, candidates, candidate_scores=None, reference_scores=None):
        candidate_scores = candidate_scores if candidate_scores is not None else self.evaluate(candidates)[1]
        reference_scores = reference_scores if reference_scores is not None else self.references_agreement()[1]
        with open(path, 'w') as f:
            f.write(self.get_name(detailed=True) + '\n')
            f.write('candidates score: {:.3f} (mean)\n'.format(mean(candidate_scores)))
            f.write('-----------------------------------------------\n')
            f.write('references agreement: {:.3f} (mean)\n\n'.format(mean(reference_scores)))
            f.write('candidates score: {:.3f} (harmonic mean)\n'.format(harmonic_mean(candidate_scores)))
            f.write('references agreement: {:.3f} (harmonic mean)\n\n'.format(harmonic_mean(reference_scores)))
            f.write('candidates score: {:.3f} (aggregate)\n'.format(self.aggregate_scores(candidate_scores)))
            f.write('references agreement: {:.3f} (aggregate)\n'.format(self.aggregate_scores(reference_scores)))
            f.write('-----------------------------------------------\n')
            for key, _candidates in candidates.items():
                for candidate, can_score, ref_score in zip(_candidates, candidate_scores[key], reference_scores[key]):
                    f.write('id: {}\n\n'.format(key))
                    f.write('candidate:\n----> {}\n'.format(candidate))
                    f.write('references:\n{}\n\n'.format('\n'.join(['===== ' + s for s in self.references[key]])))
                    f.write('candidate score: {:.3f}\n'.format(can_score))
                    f.write('reference score: {:.3f}\n'.format(ref_score))
                    f.write('-----------------------------------------------\n')
