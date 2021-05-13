import os
from formeval.utils import *
from formeval.spice import *
from formeval.cider import CiderEvaluator
from formeval.spice import SpiceEvaluator
from formeval.nca import NCAEvaluator
from formeval.bleu import BleuEvaluator



def summary():
    file_path = os.getcwd() + '/exp_data/commongen_t5_baseline_dev.jsonl'
    references = jsonl_to_dict(file_path, key='id', value='references', allow_duplicated_keys=True)
    candidates = jsonl_to_dict(file_path, key='id', value='candidate', allow_duplicated_keys=True)
    _file_path = file_path[:-6]

    evaluator = NCAEvaluator(references=references)
    evaluator.compile_report(candidates, path=_file_path + '_nca.txt')

    evaluator = CiderEvaluator(references=references)
    evaluator.compile_report(candidates, path=_file_path + '_cider.txt')


def main():
    file_path = os.getcwd() + '/exp_data/commongen_t5_baseline_test.jsonl'
    references = jsonl_to_dict(file_path, key='id', value='references', allow_duplicated_keys=False)
    candidates = jsonl_to_dict(file_path, key='id', value='candidate', allow_duplicated_keys=False)

    for v in candidates.values():
        if len(v) != 1:
            print(v)

    evaluator = SpiceEvaluator(references)
    score, scores = evaluator.evaluate(candidates)
    # score, scores = evaluator.references_agreement_by_sampling()
    print(score)


if __name__ == "__main__":
    main()
