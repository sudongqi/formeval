import nltk
from .processor import FormProcessor
from .base_evaluator import BaseEvaluator


class NLTKBleuEvaluator(BaseEvaluator):

    def __init__(self, references, processor=None, already_processed=False, silent=True,
                 weights=(0.25, 0.25, 0.25, 0.25),
                 smoothing_function=None,
                 auto_reweigh=False
                 ):
        super(NLTKBleuEvaluator, self).__init__(references=references,
                                                processor=processor if processor else FormProcessor(),
                                                already_processed=already_processed,
                                                silent=silent
                                                )
        self._references = self._process(references)
        self.weights = weights
        self.smoothing_function = smoothing_function
        self.auto_reweigh = auto_reweigh
        self.log_setup_finish()

    def evaluate(self, candidates):
        self.log_evaluation_start(candidates)
        _references, _candidates = [], []
        for key, sentences in self._process(candidates).items():
            for sentence in sentences:
                _references.append(self._references[key])
                _candidates.append(sentence)
        self.log_data_stats(candidates, 'candidates')
        score = nltk.translate.bleu_score.corpus_bleu(_references,
                                                      _candidates,
                                                      weights=self.weights,
                                                      smoothing_function=self.smoothing_function,
                                                      auto_reweigh=self.auto_reweigh
                                                      )
        self.log_evaluation_finish()
        return score, None

    def get_name(self, detailed=True):
        res = 'bleu'
        if detailed:
            res += ' (weights=[{}], auto_reweigh={})'.format(' '.join(self.weights), self.auto_reweigh)
        return res

    def compile_report(self, *args, **kwargs):
        self._log('bleu evaluation is corpus based; skip compile_report() ...')
