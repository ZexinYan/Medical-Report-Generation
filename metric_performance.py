from pycocoevalcap.eval import calculate_metrics
import numpy as np
import json
import argparse


def create_dataset(array):
    dataset = {'annotations': []}

    for i, caption in enumerate(array):
        dataset['annotations'].append({
            'image_id': i,
            'caption': caption
        })
    return dataset


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str,
                        default='./results/clean_test.json')
    args = parser.parse_args()

    test = load_json(args.result_path)
    datasetGTS = {'annotations': []}
    datasetRES = {'annotations': []}

    for i, image_id in enumerate(test):
        array = []
        for each in test[image_id]['Pred Sent']:
            array.append(test[image_id]['Pred Sent'][each])
        pred_sent = '. '.join(array)

        array = []
        for each in test[image_id]['Real Sent']:
            sent = test[image_id]['Real Sent'][each]
            if len(sent) != 0:
                array.append(sent)
        real_sent = '. '.join(array)
        datasetGTS['annotations'].append({
            'image_id': i,
            'caption': real_sent
        })
        datasetRES['annotations'].append({
            'image_id': i,
            'caption': pred_sent
        })

    rng = range(len(test))
    print calculate_metrics(rng, datasetGTS, datasetRES)
