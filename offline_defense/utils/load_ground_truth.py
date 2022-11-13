#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load ground truth of a dataset.
As part of 11-785 Course Project
Author: Ronghao Ni (ronghaon)
Last revised Nov. 2022
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import argparse

parser = argparse.ArgumentParser(description='Load ground truth of a dataset.')

parser.add_argument('-d', '--dataset', default='librispeech',
                    help='Dataset to load. Default: librispeech')

parser.add_argument('-s', '--split', default='test_clean',
                    help='Split of dataset to load. Default: test_clean')

parser.add_argument('-p', '--path', default='/data/armory_datasets/',
                    help='Path to dataset. If this is not specified, program will try to download the dataset and load')

parser.add_argument('-n', '--number', default='100',
                    help='Number of groud truth to fetch. Default: 100')

parser.add_argument('dumpfile', default='gt.pkl',
                    help='where to dump the ground truth to')

args = parser.parse_args()


def load_ground_truth(dataset='librispeech', split='test_clean', data_dir=None, number_to_fetch=100):
    """Load ground truth from dataset

    Args:
        dataset (str, optional): Dataset to load. Defaults to 'librispeech'.
        split (str, optional): Split of dataset to load. Defaults to 'test_clean'.
        data_dir (str, optional): Path to dataset. Defaults to None.
        number_to_fetch (int, optional): Number of groud truth to fetch. Defaults to 100.

    Returns:
        List: a list of ground truth strings
    """
    ds = tfds.load(dataset, split=split, data_dir=data_dir,
                   download=False if data_dir else True)
    ds = tfds.as_numpy(ds)
    index = 0
    strings = []
    for data in ds:
        if index == number_to_fetch:
            break
        strings.append(data['text'])
        index += 1
    return strings


def dump_to_file(filepath, obj):
    """Dump object to file

    Args:
        filepath (str): where to dump the ground truth t
        obj (Object): Object to dump
    """
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)


if __name__ == "__main__":
    strings = load_ground_truth(dataset=args.dataset, split=args.split,
                                data_dir=args.path, number_to_fetch=int(args.number))
    dump_to_file(args.dumpfile, strings)
