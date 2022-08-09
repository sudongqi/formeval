import os
from .bleu import BleuEvaluator
from .cider import CiderEvaluator
from .nca import NCAEvaluator
from .spice import SpiceEvaluator
from .utils import *

NAME_TO_EVALUATOR = {
    'bleu': BleuEvaluator,
    'cider': CiderEvaluator,
    'nca': NCAEvaluator,
    'spice': SpiceEvaluator
}


def evaluate_multiple_files(paths,
                            evaluators,
                            id_key='id',
                            candidate_key='candidate',
                            ref_paths=None,
                            references_key='references',
                            self_agreement_n_samples=5,
                            ):
    if ref_paths is None:
        log('entered self-agreement mode')
        _paths = zip(paths, paths)
        references_key = candidate_key
    else:
        _paths = zip(paths, ref_paths)

    for pred_path, ref_path in _paths:
        path_head, path_tail = os.path.split(pred_path)
        candidates = jsonl_to_dict(pred_path,
                                   key=id_key,
                                   value=candidate_key)
        references = jsonl_to_dict(ref_path,
                                   key=id_key,
                                   value=references_key)

        evaluators = {name: NAME_TO_EVALUATOR[name](references) for name in evaluators}
        for name, evaluator in evaluators.items():
            with enclose_timer('evaluating {} with [{}] ...'.format(pred_path, name)):
                res = evaluator.compile_report(path=path_join(path_head, path_tail) + '.{}.report'.format(name),
                                               candidates=candidates, self_agreement_n_samples=self_agreement_n_samples)
                log(res)
