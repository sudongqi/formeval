import nltk
from .utils import UnknownCandidatesError
from .processor import RegexProcessor
from .evaluator import FormEvaluator


class BleuEvaluator(FormEvaluator):

    def __init__(self, references, processor=None, already_processed=False, silent=True):
        super(BleuEvaluator, self).__init__(silent=silent,
                                            processor=processor if processor else RegexProcessor(),
                                            already_processed=already_processed
                                            )
        self.log_data_stats(references, 'references')
        self.references = self._process(references)
        self.log_build_references_completed()

    def evaluate(self, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False):
        self.log_evaluation_started(candidates)
        _references, _candidates = [], []
        for key, sentences in self._process(candidates).items():
            for sentence in sentences:
                if key in self.references:
                    _references.append(self.references[key])
                    _candidates.append(sentence)
                else:
                    raise UnknownCandidatesError
        self.log_data_stats(candidates, 'candidates')
        score = nltk.translate.bleu_score.corpus_bleu(_references,
                                                      _candidates,
                                                      weights=weights,
                                                      smoothing_function=smoothing_function,
                                                      auto_reweigh=auto_reweigh
                                                      )
        self.log_evaluation_completed()
        return score, None
