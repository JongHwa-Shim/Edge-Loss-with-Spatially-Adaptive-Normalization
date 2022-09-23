from os.path import join

from dataset import DatasetFromFolder, DatasetFromFolderCelebA


def get_training_set(root_dir, direction, opt):
    train_dir = join(root_dir, "train")
    if opt.dataset == "celeba":
        return DatasetFromFolderCelebA(train_dir, direction, opt)
    else:
        return DatasetFromFolder(train_dir, direction, opt)


def get_test_set(root_dir, direction, opt):
    test_dir = join(root_dir, "test")
    if opt.dataset == "celeba":
        return DatasetFromFolderCelebA(test_dir, direction, opt)
    else:
        return DatasetFromFolder(test_dir, direction, opt)