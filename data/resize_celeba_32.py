import logging
import os
import sys
import numpy as np
from torchvision.utils import save_image

import data_loaders as dl
from torch.utils.data import DataLoader


os.chdir('../')

def make_directory_if_not_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class create_train_test(object):
    def __init__(self, dataset_name="mnist"):

        self._dataset_name = dataset_name.lower()

        if self._dataset_name=='celeba':
            self._dataset = dl.CelebAImageReader(path="datasets/celeba", batch_size=64, setname='train')

        else:
            print("Visualize Data Class Init Error dataloaders")
            raise NotImplementedError

        self._dataset_path = os.getcwd()+"/datasets2"#"/".join(os.getcwd().split("/")[:-1])
        make_directory_if_not_exists(self._dataset_path)

        self._dataset_path += f"/{self._dataset_name}"
        print("Dataset Path: ", self._dataset_path)
        make_directory_if_not_exists(self._dataset_path)

    def tr_test_copy(self):
        print("Reorganize train...")
        #Train Folder
        train_dataset = dl.CelebAImageReader(path="datasets/celeba", batch_size=64, setname='train')
        self._train_dataset = DataLoader(train_dataset,batch_size=32,shuffle=False)

        train_path = self._dataset_path + "/" + "train"
        make_directory_if_not_exists(train_path)

        counter=0
        for index, (images, labels) in enumerate(self._train_dataset):
            # print('index: ', index, images.shape, labels)
            for image, label in zip(images,labels):
                counter += 1
                class_path = train_path + "/" + str(label.numpy())
                make_directory_if_not_exists(class_path)
                save_image(image,class_path+f"/{counter}_{label.numpy()}.png")
            print("counter: ", counter)
        print("Train Done")

        print("Reorganize val...")
        # Train Folder
        val_dataset = dl.CelebAImageReader(path="datasets/celeba", batch_size=64, setname='val')
        self._val_dataset = DataLoader(val_dataset, batch_size=32, shuffle=False)

        val_path = self._dataset_path + "/" + "train"
        make_directory_if_not_exists(val_path)

        counter = 0
        for index, (images, labels) in enumerate(self._val_dataset):
            for image, label in zip(images, labels):
                counter += 1
                class_path = val_path + "/" + str(label.numpy())
                make_directory_if_not_exists(class_path)
                save_image(image, class_path + f"/{counter}_{label.numpy()}.png")
            print("counter: ", counter)
        print("Val Done")



        print("Reorganize test ...")
        test_dataset = dl.CelebAImageReader(path="datasets/celeba", batch_size=64, setname='test')
        self._test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_path = self._dataset_path + "/" + "test"
        make_directory_if_not_exists(test_path)
        counter=0
        for index, (images, labels) in enumerate(self._test_dataset):
            for image, label in zip(images,labels):
                counter += 1
                class_path = test_path + "/" + str(label.numpy())
                make_directory_if_not_exists(class_path)
                save_image(image,class_path+f"/{counter}_{label.numpy()}.png")
            print("counter: ", counter)

        print("Test Done")



reorganize = create_train_test("celeba")
reorganize.tr_test_copy()
print("Need to do for CelebA Dataset (torchvision error). Download, manually split the dataset into train,test,val"
      "and place in datasets")

