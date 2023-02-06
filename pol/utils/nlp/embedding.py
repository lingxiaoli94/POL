'''
Tools to process XLM data found in
http://manikvarma.org/downloads/XC/XMLRepository.html
'''

import gzip
import json
import h5py
import numpy as np
import torch
import argparse
import tqdm
from sentence_transformers import SentenceTransformer

class TextFeatureExtractor:
    def __init__(self,
                 model_name='all-mpnet-base-v2',
                 h5_compression_params={
                     'compression': 'gzip',
                     'compression_opts': 9
                 }):
        self.model = SentenceTransformer(model_name)
        self.h5_compression_params = h5_compression_params

    def process_raw(self, h5_gp, data_json_gz, has_label):
        contents = [
            json.loads(line.decode('utf-8').rstrip('\n'))
            for line in gzip.open(data_json_gz, 'r')]
        num_point = len(contents)
        titles = [c['title'] for c in contents]
        # Encode each title into a unit-norm feature vector.
        title_features = np.array([self.model.encode(t) for t in tqdm.tqdm(
            titles, desc='Extracting title ')])
        h5_gp.create_dataset('point_features', data=title_features,
                                 **self.h5_compression_params)
        contexts = [c['content'] for c in contents]
        context_features = np.array([self.model.encode(t) for t in tqdm.tqdm(
            contexts, desc='Extracting context ')])
        h5_gp.create_dataset('context_features', data=context_features,
                                 **self.h5_compression_params)

        if not has_label:
            return
        # HACK: instead of using a sparse representation, we use a dense
        # matrix to store the labels for each point.
        max_num_label = np.array([len(c['target_ind']) for c in contents]).max()
        assert(max_num_label < 100)
        labels = []
        for i in range(num_point):
            inds = contents[i]['target_ind']
            l = np.full([max_num_label], -1, dtype=int)
            for j in range(len(inds)):
                l[j] = inds[j]
            labels.append(l)
        labels = np.array(labels)
        h5_gp.create_dataset('labels', data=labels,
                                 **self.h5_compression_params)

    def process_all(self, train_json_gz, test_json_gz, label_json_gz,
                    h5_out):
        h5_handle = h5py.File(h5_out, 'w')

        train_gp = h5_handle.create_group('train')
        test_gp = h5_handle.create_group('test')
        label_gp = h5_handle.create_group('label')
        self.process_raw(train_gp, train_json_gz, True)
        self.process_raw(test_gp, test_json_gz, True)
        self.process_raw(label_gp, label_json_gz, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_json_gz', type=str)
    parser.add_argument('test_json_gz', type=str)
    parser.add_argument('label_json_gz', type=str)
    parser.add_argument('h5_out', type=str)
    args = parser.parse_args()

    extractor = TextFeatureExtractor()
    extractor.process_all(args.train_json_gz,
                          args.test_json_gz,
                          args.label_json_gz,
                          args.h5_out)
