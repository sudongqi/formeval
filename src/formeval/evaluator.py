from .utils import sample_from_references, SimpleTimer


def self_agreement_score(references, evaluator, num_trials=10, discard_identical=True, **kwargs):
    res = []
    for i in range(num_trials):
        _candidates, _references = sample_from_references(references, discard_identical=discard_identical)
        _evaluator = evaluator(references=_references, **kwargs)
        res.append(_evaluator.evaluate(_candidates)[0])
    return sum(res) / len(res)


class FormEvaluator:

    def __init__(self, silent, processor, already_processed):
        self.silent = silent
        self.processor = processor
        self.already_processed = already_processed
        self.timer = SimpleTimer()

    def log(self, message):
        if not self.silent:
            print(message)

    def _process(self, data):
        self.timer.check()
        res = self.processor.process(data) if not self.already_processed else data
        self.log('processor.process(data) took {:3f}s'.format(self.timer.check()))
        return res

    def log_data_stats(self, data, name='data'):
        self.log('{} has {} unique ids and {} sentences'.format(name, len(data), sum(map(len, data.values()))))

    def log_build_references_started(self, references):
        self.log_data_stats(references, 'references')

    def log_build_references_completed(self):
        self.log('references processing completed, took {:.3f}s'.format(self.timer.total_time()))

    def log_evaluation_started(self, candidates):
        self.timer = SimpleTimer()
        self.log_data_stats(candidates, 'candidates')

    def log_evaluation_completed(self):
        self.log('candidates evaluation completed, took {:.3f}s'.format(self.timer.total_time()))
