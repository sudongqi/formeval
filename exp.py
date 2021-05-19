import os
from formeval.utils import *
from formeval.spice import *
from formeval.evaluator import evaluate_multiple_files
from formeval.spice import SpiceEvaluator
from formeval.cider import CiderEvaluator
from formeval.nltk_bleu import NLTKBleuEvaluator
from formeval.bleu import BleuEvaluator


def summary():
    paths = [os.getcwd() + p for p in ['/exp_data/t5small_cg_dev.jsonl',
                                       '/exp_data/t5small_cg_test.jsonl',
                                       '/exp_data/t5small_4to1_cg_dev.jsonl',
                                       '/exp_data/t5small_4to1_cg_test.jsonl',
                                       '/exp_data/t5small_10to1_gcc_snli_cg_dev.jsonl',
                                       '/exp_data/t5small_10to1_gcc_snli_cg_test.jsonl'
                                       ]]

    evaluate_multiple_files(paths=paths,
                            evaluator_names=['bleu', 'cider', 'spice', 'nca'],
                            id_key='src',
                            candidate_key='pred',
                            references_key='tar'
                            )


def main():
    working_dir = os.getcwd()
    references = jsonl_to_dict(working_dir + '/exp_data/t5small_cg_test.jsonl', key='src', value='tar')
    candidates = jsonl_to_dict(working_dir + '/exp_data/t5small_cg_test.jsonl', key='src', value='pred')
    evaluator = SpiceEvaluator(references=references, silent=False)
    score, scores = evaluator.evaluate(candidates)
    print(score)
    evaluator = BleuEvaluator(references=references, silent=False)
    score, scores = evaluator.evaluate(candidates)
    print(score)


if __name__ == "__main__":
    summary()
    # main()