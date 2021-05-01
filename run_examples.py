import os
from formeval.cider import CiderEvaluator
from formeval.utils import jsonl_iter, jsonl_to_dict
from formeval.processor import RegexProcessor


def cider_evaluate_pascal():
    # references {'id1': ['sentence_1', 'sentence_2'], 'id2': ['sentence_3', 'sentence_4', ...], ...}
    references = jsonl_to_dict(jsonl_iter(cwd + '/example_data/pascal_ref.jsonl'))
    # candidates has same format as references
    candidates = jsonl_to_dict(jsonl_iter(cwd + '/example_data/pascal_pred.jsonl'))
    # processor.process(references) = {'id1': [['token_1', 'token_2', ...], ['token_1', 'token_2', ...]], 'id2': ...}
    # processor is optional
    processor = RegexProcessor()
    evaluator = CiderEvaluator(references=references, processor=processor)
    score = evaluator.evaluate(candidates)
    print("pascal cider score: {}".format(score))


if __name__ == "__main__":
    cwd = os.getcwd()
    cider_evaluate_pascal()
