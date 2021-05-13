import nltk
from .utils import UnknownCandidatesError
from .processor import FormProcessor
from .evaluator import FormEvaluator


class BleuEvaluator(FormEvaluator):

    def __init__(self, references, processor=None, already_processed=False, silent=True,
                 weights=(0.25, 0.25, 0.25, 0.25),
                 smoothing_function=None,
                 auto_reweigh=False
                 ):
        super(BleuEvaluator, self).__init__(references=references,
                                            processor=processor if processor else FormProcessor(),
                                            already_processed=already_processed,
                                            silent=silent
                                            )
        self.tokenized_references = self._process(references)
        self.weights = weights
        self.smoothing_function = smoothing_function
        self.auto_reweigh = auto_reweigh
        self.log_build_references_timer()

    def evaluate(self, candidates):
        self.check_candidates_and_start_evaluation_timer(candidates)
        _references, _candidates = [], []
        for key, sentences in self._process(candidates).items():
            for sentence in sentences:
                if key in self.tokenized_references:
                    _references.append(self.tokenized_references[key])
                    _candidates.append(sentence)
                else:
                    raise UnknownCandidatesError
        self.log_data_stats(candidates, 'candidates')
        score = nltk.translate.bleu_score.corpus_bleu(_references,
                                                      _candidates,
                                                      weights=self.weights,
                                                      smoothing_function=self.smoothing_function,
                                                      auto_reweigh=self.auto_reweigh
                                                      )
        self.log_evaluation_timer()
        return score, None

    def get_name(self, detailed):
        res = 'bleu'
        if detailed:
            res += ' (weights=[{}], auto_reweigh={})'.format(' '.join(self.weights), self.auto_reweigh)
        return res
