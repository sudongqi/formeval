import collections
import subprocess
import os
import json
import tempfile
from ._evaluator import BaseEvaluator
from .processor import FormProcessor

TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'


class SpiceEvaluator(BaseEvaluator):
    def __init__(self, references, processor=None, already_processed=False, silent=True):
        super(SpiceEvaluator, self).__init__(references=references,
                                             processor=processor if processor else FormProcessor(),
                                             already_processed=already_processed,
                                             silent=silent
                                             )

    def evaluate(self, candidates):

        input_data = []
        for key, _candidate in candidates.items():
            for candidate in _candidate:
                input_data.append({
                    "image_id": key,
                    "test": candidate,
                    "refs": self.references[key]
                })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=temp_dir)
        json.dump(input_data, in_file)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir = os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        spice_jar = os.path.join(cwd, 'spice_jar/spice-1.0.jar')
        spice_cmd = ['java', '-jar', '-Xmx8G', spice_jar, in_file.name,
                     '-cache', cache_dir,
                     '-out', out_file.name,
                     '-subset',
                     '-silent'
                     ]
        subprocess.check_call(spice_cmd, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=subprocess.DEVNULL)

        # Read and process results
        with open(out_file.name) as data_file:
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        scores = collections.defaultdict(list)
        for item in results:
            scores[item['image_id']].append(item['scores']['All']['f'])
        return self.aggregate_scores(scores), scores

    def get_name(self, detailed=True):
        return 'spice'
