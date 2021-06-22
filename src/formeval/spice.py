import collections
import subprocess
import os
import json
import tempfile
from .base_evaluator import BaseEvaluator
from .processor import FormProcessor
from .utils import root_dir

TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'
JAR_PATH = 'spice_dependencies/spice-1.0.jar'


class SpiceEvaluator(BaseEvaluator):
    def __init__(self, references, processor=None, already_processed=False, silent=True,
                 jar_path=None,
                 temp_dir=None,
                 cache_dir=None):
        super(SpiceEvaluator, self).__init__(references=references,
                                             processor=processor if processor else FormProcessor(),
                                             already_processed=already_processed,
                                             silent=silent,
                                             )

        self.temp_dir = os.path.join(root_dir(), TEMP_DIR) if temp_dir is None else temp_dir
        self.cache_dir = os.path.join(root_dir(), CACHE_DIR) if cache_dir is None else cache_dir
        self.spice_jar = os.path.join(root_dir(), JAR_PATH) if jar_path is None else jar_path
        self.log_setup_finish()

    def evaluate(self, candidates):
        self.log_evaluation_start(candidates)
        input_data = []
        for key, _candidate in candidates.items():
            for candidate in _candidate:
                input_data.append({
                    "image_id": key,
                    "test": candidate,
                    "refs": self.references[key]
                })

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        in_file = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=self.temp_dir)
        json.dump(input_data, in_file)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir)
        out_file.close()
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        spice_cmd = ['java', '-jar', '-Xmx8G', self.spice_jar, in_file.name,
                     '-cache', self.cache_dir,
                     '-out', out_file.name,
                     '-subset',
                     '-silent'
                     ]
        subprocess.check_call(spice_cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL
                              )

        # Read and process results
        with open(out_file.name) as data_file:
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        scores = collections.defaultdict(list)
        for item in results:
            scores[item['image_id']].append(item['scores']['All']['f'])
        self.log_evaluation_finish()
        return self.aggregate_scores(scores), scores

    def get_name(self, detailed=True):
        return 'spice'
