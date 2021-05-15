import os
from formeval.utils import *
from formeval.spice import *
from formeval.evaluator import evaluate_multiple_files
from formeval.spice import SpiceEvaluator
from formeval.cider import CiderEvaluator
from formeval.bleu import BleuEvaluator


def summary():
    paths = [os.getcwd() + p for p in ['/exp_data/t5base_dev.jsonl',
                                       '/exp_data/t5base_test.jsonl',
                                       '/exp_data/t5base_4to1_dev.jsonl',
                                       '/exp_data/t5base_4to1_test.jsonl'
                                       ]]

    evaluate_multiple_files(paths=paths, evaluator_names=['bleu', 'cider', 'spice', 'nca'])


def main():
    pass


if __name__ == "__main__":
    summary()
    # main()
