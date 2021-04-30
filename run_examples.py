import os
from formeval.cider import CiderEvaluator
from formeval.utils import jsonl_iter, jsonl_to_dict
from formeval.processor import RegexProcessor


def cider_evaluate_pascal():
    references = jsonl_to_dict(jsonl_iter(cwd + '/example_data/pascal_ref.jsonl'))
    candidates = jsonl_to_dict(jsonl_iter(cwd + '/example_data/pascal_pred.jsonl'))
    # references {'id1': ['sentence_1', 'sentence_2'], 'id2': ['sentence_3', 'sentence_4', ...], ...}
    # processor.process(references) = {'id1': [['token_1', 'token_2', ...], ['token_1', 'token_2', ...]], 'id2': ...}
    evaluator = CiderEvaluator(references=references, processor=RegexProcessor())
    # candidates has same format as references
    score = evaluator.evaluate(candidates)
    print("pascal cider score: {}".format(score))


if __name__ == "__main__":
    cwd = os.getcwd()
    cider_evaluate_pascal()
