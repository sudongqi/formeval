import os
from formeval.cider import CiderEvaluator
from formeval.bleu import BleuEvaluator
from formeval.utils import jsonl_to_dict
from formeval.processor import FormProcessor, regexp_tokenizer


def cider_self_agreement_pascal():
    working_dir = os.getcwd()
    references = jsonl_to_dict(working_dir + '/example_data/pascal_ref.jsonl')
    score, _ = CiderEvaluator(references).references_agreement_by_sampling(silent=False)
    print('pascal references cider agreement score: {}\n'.format(score))


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
    processor = FormProcessor(tokenizer=regexp_tokenizer(), lemmatizer=None)
    evaluator = CiderEvaluator(references=references, processor=processor, silent=False)
    score, scores = evaluator.evaluate(candidates)
    print("pascal example cider score: {}\n".format(score))


def bleu_evaluate_pascal():
    working_dir = os.getcwd()
    references = jsonl_to_dict(working_dir + '/example_data/pascal_ref.jsonl')
    candidates = jsonl_to_dict(working_dir + '/example_data/pascal_pred.jsonl')
    evaluator = BleuEvaluator(references=references, silent=False)
    score, _ = evaluator.evaluate(candidates)
    print("pascal example bleu score: {}\n".format(score))


if __name__ == "__main__":
    cider_evaluate_pascal()
    bleu_evaluate_pascal()
    cider_self_agreement_pascal()
