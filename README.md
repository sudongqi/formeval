# Language Form Evaluation

This package includes efficient pure python implementation of the following metrics:

* [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) (in progress)
* [CIDEr/CIDEr-D](https://arxiv.org/pdf/1411.5726.pdf)
* [SPICE](https://arxiv.org/pdf/1607.08822.pdf) (in progress)

## Dependencies

* python3.6 +
* nltk

## Setup

    pip install formeval
    pip install nltk

optional setup

    python3 -c 'import nltk; nltk.download("wordnet")'

## Example Data

* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
* [CommonGen](https://github.com/INK-USC/CommonGen)
    

## Usage

See [run_examples.py](https://github.com/sudongqi/lfeval/blob/main/run_examples.py)