from os.path import join

from dataset import DatasetFromFolder


def get_training_set(root_dir, direction, opt):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir, direction, opt)


def get_test_set(root_dir, direction, opt):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir, direction, opt)
