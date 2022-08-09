from formeval.cider import CiderEvaluator
from formeval.bleu import BleuEvaluator
from formeval.processor import FormProcessor, regexp_tokenizer
from formeval.utils import *
from formeval.evaluator import evaluate_multiple_files


def main():
    # {'id1': ['sentence_1', 'sentence_2'], 'id2': ['sentence_3', 'sentence_4', ...], ...}
    candidates = jsonl_to_dict(this_dir() + '/example_data/pascal_pred.jsonl')
    references = jsonl_to_dict(this_dir() + '/example_data/pascal_ref.jsonl')

    # bleu score
    with enclose_timer('bleu score'):
        evaluator = BleuEvaluator(references=references, silent=False)
        score, _ = evaluator.evaluate(candidates)
        log("pascal example bleu score: {}".format(score))

    # cider score
    with enclose_timer('cider score'):
        processor = FormProcessor(tokenizer=regexp_tokenizer(), lemmatizer=None)
        evaluator = CiderEvaluator(references=references, processor=processor, silent=False)
        score, scores = evaluator.evaluate(candidates)
        log("pascal example cider score: {}".format(score))

    # one-liner self agreement evaluation report
    with enclose_timer('complete evaluations with multiple evaluator'):
        evaluate_multiple_files(paths=[this_dir() + '/example_data/pascal_pred.jsonl'],
                                ref_paths=[this_dir() + '/example_data/pascal_ref.jsonl'],
                                evaluators=['bleu', 'cider', 'nca'],
                                id_key='id',
                                candidate_key='sentence',
                                references_key='sentence',
                                )


if __name__ == "__main__":
    main()
