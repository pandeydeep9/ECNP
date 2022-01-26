import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


idx_counter = torch.tensor([0])
idx_counter.share_memory_()

def get_idx_counter():
    global idx_counter
    idx_counter += 1
    if idx_counter>1e6:
        idx_counter=0
    return idx_counter

class MNISTOneClass(object):
    def __init__(self, setname, x_width=28, y_height=28, classname= '5'):
        self._x_width = x_width
        self._y_height = y_height

        self._data_path = os.getcwd()+"/datasets"

        setname = setname.lower()
        path = get_setname_path(setname,self._data_path, "mnist")

        self.data= get_data_specified_class(path,classname)

        self.transform = transforms.Compose(
            [
                transforms.Resize((self._x_width, self._y_height)),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample_path = self.data[idx]
        sample = Image.open(sample_path)
        label = sample_path.split('/')[-2]
        # print(label)

        sample = self.transform(sample)
        # print("mnist: ", sample.shape)
        return sample, torch.tensor(int(label))


class MNISTImageLoader(object):
    def __init__(self, setname, x_width=28, y_height=28):
        self._x_width = x_width
        self._y_height = y_height

        self._data_path = os.getcwd()+"/datasets"

        setname = setname.lower()
        path = get_setname_path(setname,self._data_path, "mnist")

        self.data= get_data_all(path)

        self.transform = transforms.Compose(
            [
                transforms.Resize((self._x_width, self._y_height)),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample_path = self.data[idx]
        sample = Image.open(sample_path)
        label = sample_path.split('/')[-2]
        # print(label)

        sample = self.transform(sample)
        # print("mnist: ", sample.shape)
        return sample, torch.tensor(int(label))




class Cifar10ImageLoader(object):
    def __init__(self, setname,  path =os.getcwd()+"/datasets", x_width=32, y_height=32):
        self._x_width = x_width
        self._y_height = y_height

        self._data_path = path
        setname = setname.lower()

        path = get_setname_path(setname, self._data_path, "cifar10")

        self.data = get_data_all(path)
        print(f"len: {setname}", len(self.data))

        self.transform = transforms.Compose(
            [
                transforms.Resize((self._x_width,self._y_height)),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_path = self.data[idx]
        sample = Image.open(sample_path)

        sample = self.transform(sample)
        return sample, 1

class CelebAImageLoader(object):
    def __init__(self, setname,  path =os.getcwd()+"/datasets", crop=89,x_width=32, y_height=32):
        self._x_width = x_width
        self._y_height = y_height

        self._data_path = path
        setname = setname.lower()

        path = get_setname_path(setname, self._data_path, "celeba")

        self.data = get_data_all(path)
        print(f"len: {setname}", len(self.data))

        self.transform = transforms.Compose(
            [
                # transforms.CenterCrop(crop),
                transforms.Resize((self._x_width,self._y_height)),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_path = self.data[idx]
        sample = Image.open(sample_path)

        sample = self.transform(sample)
        return sample, 0

def get_setname_path(setname, data_path, dataset):
    if setname not in ['train','test','val']:
        raise ValueError("Wrong set name passed to Multi dataset Image Loader")

    path = os.path.join(data_path + "/" + dataset + "/" + setname)
    return path

def get_data_all(setpath):
    data = []
    # print("setpath: ", setpath)
    for class_name in os.listdir(setpath):
        class_path = os.path.join(setpath, class_name)
        # print(class_path)
        for image_name in os.listdir(class_path):
            data.append(os.path.join(class_path, image_name))
    return data

def get_data_specified_class(setpath, name_of_class = '5'):
    data = []
    # print("setpath: ", setpath)
    class_name = name_of_class
    class_path = os.path.join(setpath, class_name)
    # print(class_path)
    for image_name in os.listdir(class_path):
        data.append(os.path.join(class_path, image_name))
    return data



class VisualizeData(object):
    def __init__(self, dataset_name="mnist"):
        self._dataset_name = dataset_name.lower()

        if self._dataset_name=='mnist':
            self._dataset = MNISTImageLoader(batch_size=16).train_loader
        elif self._dataset_name=='cifar10':
            self._dataset = Cifar10ImageLoader(batch_size=16).train_loader
        elif self._dataset_name=='celeba':
            self._dataset = CelebAImageLoader(path="datafiles/celeba", batch_size=64, setname='train')
        elif self._dataset_name=='multi':
            self._dataset = MultiDatasetImageLoader(setname='train',dataset_1="celeba", dataset_2="cifar10")

        else:
            print("Visualize Data Class Init Error dataloaders")
            raise NotImplementedError

    def visualize_data(self):
        print("Okay Let's Visualize the data")
        images,labels = iter(self._dataset).next()

        batch_size = images.shape[0]


        for index, (image,label) in enumerate(zip(images,labels)):
            if len(image.shape)==3:
                image = image.permute(1,2,0).numpy()*255
            elif len(images.shape)==2:
                print("shape of image for some reason is 2(visualize data, dataloaders)")
                raise NotImplementedError
            else:
                print("Something is not correct(visualize data, dataloaders)")
                raise NotImplementedError
            plt_idx = index
            plt.subplot(np.sqrt(batch_size)+1,np.sqrt(batch_size)+1,index+1)
            if self._dataset_name == 'mnist':
                image = np.tile(image,(1,1,3))
            print("image shape: ", image.shape)
            plt.imshow(image.astype('uint8'))
            plt.axis('off')
            plt.title(f"{label}",color='green',fontsize=18)
        plt.show()
        print("Done")

    def visualize_data_from_dataloader(self, dataloader):
        print("Okay Let's Visualize the data")
        images,labels = iter(dataloader).next()

        batch_size = images.shape[0]


        for index, (image,label) in enumerate(zip(images,labels)):
            if len(image.shape)==3:
                image = image.permute(1,2,0).numpy()*255
            elif len(images.shape)==2:
                print("shape of image for some reason is 2(visualize data, dataloaders)")
                raise NotImplementedError
            else:
                print("Something is not correct(visualize data, dataloaders)")
                raise NotImplementedError
            plt_idx = index
            plt.subplot(np.sqrt(batch_size)+1,np.sqrt(batch_size)+1,index+1)
            if self._dataset_name == 'mnist' or image.shape[-1]==1:
                image = np.tile(image,(1,1,3))
            print("image shape: ", image.shape)
            plt.imshow(image.astype('uint8'))
            plt.axis('off')
            plt.title(f"{label}",color='green',fontsize=18)
        plt.show()
        print("Done")
