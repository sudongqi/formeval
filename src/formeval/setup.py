import nltk
import os
from spacy.cli.download import download
from .utils import root_dir
import requests


def setup_everything():
    setup_wordnet()
    setup_spacy()
    setup_spice()


def setup_spacy():
    print('setting up spacy')
    download('en_core_web_lg')


def setup_wordnet():
    print('setting up wordnet')
    nltk.download('wordnet')


def setup_spice():
    print('setting up spice')
    prefix = 'https://github.com/sudongqi/formeval/tree/main/spice_dependencies/'
    spice_folder = 'spice_dependencies'
    lib_folder = 'lib'
    spice_jar = 'spice-1.0.jar'
    other_jars = ['Meteor-1.5.jar', 'hamcrest-core-1.3.jar', 'lmdbjni-0.4.6.jar', 'slf4j-api-1.7.12.jar',
                  'SceneGraphParser-1.0.jar', 'jackson-core-2.5.3.jar', 'lmdbjni-linux64-0.4.6.jar',
                  'slf4j-simple-1.7.21.jar', 'ejml-0.23.jar', 'javassist-3.19.0-GA.jar', 'lmdbjni-osx64-0.4.6.jar',
                  'stanford-corenlp-3.6.0-models.jar', 'fst-2.47.jar', 'json-simple-1.1.1.jar',
                  'lmdbjni-win64-0.4.6.jar', 'stanford-corenlp-3.6.0.jar', 'guava-19.0.jar', 'junit-4.12.jar',
                  'objenesis-2.4.jar']
    other_jars = [lib_folder + '/' + jar for jar in other_jars]

    os.makedirs(os.path.join(root_dir(), spice_folder, lib_folder), exist_ok=True)
    for rel_path in [spice_jar] + other_jars:
        r = requests.get(prefix + rel_path, allow_redirects=True)
        open(os.path.join(root_dir(), spice_folder, rel_path), 'wb').write(r.content)
        print('{} downloaded'.format(rel_path))
