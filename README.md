# Language Form Evaluation

This package includes efficient pure python implementation of the following metrics:

* [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) (in progress)
* [ROUGE](https://www.aclweb.org/anthology/W04-1013.pdf) (in progress)  
* [CIDEr/CIDEr-D](https://arxiv.org/pdf/1411.5726.pdf)
    * compared to original implementation: https://github.com/vrama91/cider
      * With the same tokenizer
        * error: < 1 %
        * speed: + 181 %
      * With different tokenizer (formeval cider use Regex from nltk)
        * error: ~15 %
        * speed: + 432 %
  
* [SPICE](https://arxiv.org/pdf/1607.08822.pdf) (in progress)

All stats represented above are rough estimation

## Dependencies

* python 3.6 +
* nltk 3.5+

## Setup

    pip install formeval

optional setup

    python3 -c 'import nltk; nltk.download("wordnet")'

## Example Data

* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
    

## Comparison
CIDEr/CIDEr-D (https://github.com/vrama91/cider):
* difference < 1%
* efficiency 

## Usage

See [run_examples.py](https://github.com/sudongqi/lfeval/blob/main/run_examples.py)