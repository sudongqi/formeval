from formeval.utils import *
from formeval.processor import *
from formeval.nca import *


def main():
    lemmatizer = lemminfect_lemmatizer()
    processor = FormProcessor(lemmatizer=lemmatizer)

    concept_size = []
    scores = []
    human_scores = []

    output_scores_path = os.getcwd() + '/exp_data/t5base_test.jsonl'
    write_path = output_scores_path[:-6] + '_nca.jsonl'
    with open(write_path, 'w') as w:
        for d in jsonl_iter(output_scores_path):
            id, candidate, references = d['id'], d['candidate'], d['references']

            w.write('id: {}\n'.format(id))
            w.write(summary(candidate, references))

            candidate = processor.process(candidate)
            references = processor.process(references)
            w.write(summary(candidate, references))

            can_f1 = candidate_nca_score(candidate, references)
            human_f1 = references_agreement_nca_score(references)

            concepts = get_sets_intersection(references)
            concept_size.append(len(concepts))
            w.write('concepts: {}\n'.format(concepts))
            candidate = clean(candidate, concepts)
            references = [clean(r, concepts) for r in references]
            w.write(summary(candidate, references))

            nca_can = collect_nca(candidate)
            nca_ref = collect_nca(references)
            scores.append(can_f1)
            human_scores.append(human_f1)
            w.write(
                'concept neighbors can: {}\nconcept neighbors ref: {}\nf1 score: {}\nhuman score: {}\n'.format(nca_can,
                                                                                                               nca_ref,
                                                                                                               can_f1,
                                                                                                               human_f1
                                                                                                               ))
            w.write('----------------------------\n')

        avg_score = sum(scores) / len(scores)
        human_avg_score = sum(human_scores) / len(human_scores)
        avg_concepts_size = sum(concept_size) / len(concept_size)
        w.write('avg_score: {}/{}\nhuman_baseline: {}/{}\navg_concepts_size: {}\n'.format(avg_score,
                                                                                          _harmonic_mean(scores),
                                                                                          human_avg_score,
                                                                                          _harmonic_mean(human_scores),
                                                                                          avg_concepts_size))


if __name__ == "__main__":
    main()
