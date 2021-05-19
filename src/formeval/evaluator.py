import os
from .bleu import BleuEvaluator
from .cider import CiderEvaluator
from .nca import NCAEvaluator
from .spice import SpiceEvaluator
from .utils import jsonl_to_dict, SimpleTimer

NAME_TO_EVALUATOR = {
    'bleu': BleuEvaluator,
    'cider': CiderEvaluator,
    'nca': NCAEvaluator,
    'spice': SpiceEvaluator
}


def evaluate_multiple_files(paths, evaluator_names,
                            id_key='id',
                            candidate_key='candidate',
                            references_key='references',
                            allow_duplicated_key=False):
    for path in paths:
        path_head, path_tail = os.path.split(path)
        candidates = jsonl_to_dict(path, key=id_key, value=candidate_key, allow_duplicated_keys=allow_duplicated_key)
        references = jsonl_to_dict(path, key=id_key, value=references_key, allow_duplicated_keys=allow_duplicated_key)
        evaluators = {name: NAME_TO_EVALUATOR[name](references) for name in evaluator_names}
        for name, evaluator in evaluators.items():
            print('evaluating {} with [{}] ...'.format(path, name), end='')
            timer = SimpleTimer()
            res = evaluator.compile_report(path=path_head + '/' + name + '_' + path_tail, candidates=candidates)
            print('took {:.3f} s'.format(timer.total_time()))
            print(''.join(res))
