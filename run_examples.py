import os
from formeval.cider import CiderEvaluator
from formeval.bleu import BleuEvaluator
from formeval.utils import jsonl_to_dict, combine_scores_candidates_references, write_jsonl
from formeval.processor import RegexProcessor
from formeval.evaluator import self_agreement_score


def cider_self_agreement_pascal():
    working_dir = os.getcwd()
    references = jsonl_to_dict(working_dir + '/example_data/pascal_ref.jsonl')
    score = self_agreement_score(references, CiderEvaluator, num_trials=10, processor=RegexProcessor(), silent=False)
    print('pascal references cider agreement score: {}'.format(score))


def cider_evaluate_pascal():
    """
    references (labels) is a dictionary with the following format:
        {'id1': ['sentence_1', 'sentence_2'], 'id2': ['sentence_3', 'sentence_4', ...], ...}
    candidates (predictions) has same format as references

    processor break down sentences into tokens and process (e.g. stemming) them
    processor.process(references) = {'id1': [['token_1', 'token_2', ...], ['token_1', 'token_2', ...]], 'id2': ...}
    processor is optional and customizable; every evaluator will have its own default processor
    """
    working_dir = os.getcwd()
    references = jsonl_to_dict(working_dir + '/example_data/pascal_ref.jsonl')
    candidates = jsonl_to_dict(working_dir + '/example_data/pascal_pred.jsonl')
    evaluator = CiderEvaluator(references=references, processor=RegexProcessor(), silent=False)
    score, scores = evaluator.evaluate(candidates)
    print("pascal example cider score: {}".format(score))

    predictions_with_scores = combine_scores_candidates_references(scores, candidates, references, flatten=True)
    write_jsonl(predictions_with_scores, path='./scores.jsonl')


def bleu_evaluate_pascal():
    working_dir = os.getcwd()
    references = jsonl_to_dict(working_dir + '/example_data/pascal_ref.jsonl')
    candidates = jsonl_to_dict(working_dir + '/example_data/pascal_pred.jsonl')
    evaluator = BleuEvaluator(references=references, processor=RegexProcessor(), silent=False)
    score, _ = evaluator.evaluate(candidates)
    print("pascal example bleu score: {}".format(score))


if __name__ == "__main__":
    cider_evaluate_pascal()
    # bleu_evaluate_pascal()
    # cider_self_agreement_pascal()
