import random
import os
from lfeval.cider import CiderEvaluator
from lfeval.utils import *
from lfeval.processor import *


def cider_evaluate_pascal():
    references = jsonl_to_dict(jsonl_iter(cwd + '/example_data/pascal_ref.jsonl'))
    candidates = jsonl_to_dict(jsonl_iter(cwd + '/example_data/pascal_pred.jsonl'))
    evaluator = CiderEvaluator(references, RegexWordnetProcessor())
    score = evaluator.evaluate(candidates)
    print("pascal cider score: {}".format(score))


def cider_evaluate_commongen_baseline():
    candidates, references = parse_commongen_baseline_data(cwd + '/example_data/commongen.dev.jsonl', candidate_id=0)
    evaluator = CiderEvaluator(references)
    score = evaluator.evaluate(candidates)
    print("commongen human baseline score: {}".format(score))

    random_candidate_keys = list(candidates.keys())
    random.shuffle(random_candidate_keys)
    random_candidates = {k: v for k, v in zip(random_candidate_keys, candidates.values())}
    score = evaluator.evaluate(random_candidates)
    print("commongen baseline score: {}".format(score))


if __name__ == "__main__":
    cwd = os.getcwd()
    cider_evaluate_pascal()
    cider_evaluate_commongen_baseline()
