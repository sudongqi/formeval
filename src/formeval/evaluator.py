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
                            output_dir=None,
                            self_agreement_n_samples=5,
                            ):
    log('use at most {} references for self-agreement test\n'.format(self_agreement_n_samples))

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
        if output_dir is None:
            output_dir = path_join(exec_dir(), 'formeval_report')
        for name, evaluator in evaluators.items():
            with enclose_timer('evaluating {} with [{}]'.format(pred_path, name)):
                write_path = path_join(output_dir, '{}.{}.report'.format(path_tail, name))
                make_dir(write_path)
                evaluator.compile_report(path=write_path, candidates=candidates,
                                         self_agreement_n_samples=self_agreement_n_samples)
