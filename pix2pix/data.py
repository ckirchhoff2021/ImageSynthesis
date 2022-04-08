from os.path import join

from dataset import DatasetFromFolder


def get_training_set(root_dir, direction):
    return DatasetFromFolder(root_dir, direction)


def get_test_set(root_dir, direction):
    return DatasetFromFolder(root_dir, direction)
