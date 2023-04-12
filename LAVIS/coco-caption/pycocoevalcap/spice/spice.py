from __future__ import division
import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'

class Spice:
    """
    Main Class to compute the SPICE metric 
    """

    def float_convert(self, obj):
        try:
          return float(obj)
        except:
          return np.nan

    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())
        
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id" : id,
              "test" : hypo[0],
              "refs" : ref
            })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir=os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
          os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir=os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
          os.makedirs(cache_dir)
        spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
          '-cache', cache_dir,
          '-out', out_file.name,
          '-subset',
          '-silent'
        ]
        subprocess.check_call(spice_cmd, 
            cwd=os.path.dirname(os.path.abspath(__file__)))

        # Read and process results
        with open(out_file.name) as data_file:    
          results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
          imgId_to_scores[item['image_id']] = item['scores']
          spice_scores.append(self.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in imgIds:
          # Convert none to NaN before saving scores over subcategories
          score_set = {}
          for category,score_tuple in imgId_to_scores[image_id].iteritems():
            score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
          scores.append(score_set)
        return average_score, scores

    def method(self):
        return "SPICE"


