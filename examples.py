import os
from formeval.cider import CiderEvaluator
from formeval.bleu import BleuEvaluator
from formeval.processor import FormProcessor, regexp_tokenizer
from formeval.utils import *
from formeval.evaluator import evaluate_multiple_files


def main():
    """
    references (labels) is a dictionary with the following format:
        {'id1': ['sentence_1', 'sentence_2'], 'id2': ['sentence_3', 'sentence_4', ...], ...}
    candidates (predictions) has the same format as references

    processor break down sentences into tokens and process (e.g. stemming) them
    processor.process(references) = {'id1': [['token_1', 'token_2', ...], ['token_1', 'token_2', ...]], 'id2': ...}
    processor is optional and customizable; every evaluator will have its own default processor
    """

    candidates = jsonl_to_dict(this_dir() + '/example_data/pascal_pred.jsonl')
    references = jsonl_to_dict(this_dir() + '/example_data/pascal_ref.jsonl')

    # bleu score
    evaluator = BleuEvaluator(references=references, silent=False)
    score, _ = evaluator.evaluate(candidates)
    log("pascal example bleu score: {}\n".format(score))

    # cider score
    processor = FormProcessor(tokenizer=regexp_tokenizer(), lemmatizer=None)
    evaluator = CiderEvaluator(references=references, processor=processor, silent=False)
    score, scores = evaluator.evaluate(candidates)
    log("pascal example cider score: {}\n".format(score))

    # one-liner self agreement evaluation report
    evaluate_multiple_files(paths=[this_dir() + '/example_data/pascal_pred.jsonl'],
                            ref_paths=[this_dir() + '/example_data/pascal_ref.jsonl'],
                            evaluators=['bleu', 'cider', 'nca'],
                            id_key='id',
                            candidate_key='sentence',
                            references_key='sentence',
                            )


if __name__ == "__main__":
    main()
