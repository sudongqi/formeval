# Language Form Evaluation

This package includes efficient pure python implementation of the following metrics:

* [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf)
    * current implementation is a wrapper of [nltk.translate.bleu_score](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)
      * known issue:
        * drop in performance when evaluating multi-candidates vs multi-references
* [ROUGE](https://www.aclweb.org/anthology/W04-1013.pdf) (in progress)
* [METEOR](https://www.aclweb.org/anthology/W05-0909.pdf) (in progress)
* [CIDEr/CIDEr-D](https://arxiv.org/pdf/1411.5726.pdf)
    * original implementation: https://github.com/vrama91/cider
      * with the same tokenizer
        * error: < 1 %
        * speed: + 81 %
      * with different tokenizers (FormEval use Regexp by default)
        * error: ~ 15 %
        * speed: + 332 %
  
* [SPICE](https://arxiv.org/pdf/1607.08822.pdf) (in progress)

*All stats shown above are estimations

## Dependencies

* python 3.6 +
* nltk 3.5+

## Setup

    pip install formeval

optional setup

    python3 -c 'import nltk; nltk.download("wordnet")'

## Example Data

* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

## Usage

See [run_examples.py](https://github.com/sudongqi/lfeval/blob/main/run_examples.py)