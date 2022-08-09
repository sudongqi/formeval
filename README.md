# Language Form Evaluation

This package includes efficient (almost) pure python implementation of the following metrics:

* [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf)
    * reference implementation: [nltk.translate.bleu_score](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)
        * error: < 1%
        * speed: + 189%
* [ROUGE](https://www.aclweb.org/anthology/W04-1013.pdf) (in progress)
* [METEOR](https://www.aclweb.org/anthology/W05-0909.pdf) (in progress)
* [CIDEr/CIDEr-D](https://arxiv.org/pdf/1411.5726.pdf)
    * reference implementation: https://github.com/vrama91/cider
        * with the same tokenizer
            * error: < 1 %
            * speed: + 81 %
        * with different tokenizers (FormEval use Regexp by default)
            * error: ~ 15 %
            * speed: + 332 %
* [SPICE](https://arxiv.org/pdf/1607.08822.pdf)
    * placeholder wrapper of reference implementation: https://github.com/tylin/coco-caption/tree/master/pycocoevalcap/spice
        * TODO: python scene graph parser

*All stats shown above are estimations

## Dependencies

* python 3.6 +
* nltk 3.5+

## Setup

    pip install formeval

optional setups:

* spice dependencies
* wordnet lemmatizer

    python -m formeval.setup

## Example Data

* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

## Usage

See [run_examples.py](https://github.com/sudongqi/lfeval/blob/main/run_examples.py)