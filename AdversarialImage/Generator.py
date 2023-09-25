import tensorflow as tf
import pickle
from . import Noise
import re
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


class AdversarialGenerator():

    def __init__(self,dirs=["tiny-imagenet-200/train","cifar-10-batches-py/"],res= (64,64)):
        self.dirs=dirs
        self.res=res


    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data


    def imagenet_generator(self):
        # if there is more than 1 dirrectory this function will only use the first directory
        pattern = r".*imagenet.*"
        dirs = [i for i in self.dirs if re.search(pattern, i)][0]
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.1,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        train_gen = train_datagen.flow_from_directory(directory=f"{dirs}train",
                                                      target_size=self.res,
                                                      batch_size=32,
                                                      class_mode="categorical",
                                                      seed=42,
                                                      subset="training")

        val_gen = train_datagen.flow_from_directory(directory=f"{dirs}val",
                                                    target_size=self.res,
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    seed=42,
                                                    subset="validation")

        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
        )

        test_gen = train_datagen.flow_from_directory(directory=f"{dirs}test",
                                                     target_size=self.res,
                                                     batch_size=32,
                                                     class_mode="categorical",
                                                     seed=42)

        return train_gen,val_gen,test_gen


    def __cifar_train_generator(self,dirs):
        i = 1
        while True:
            batch_data = self.unpickle(dirs + f"data_batch_{i}")
            train_images = batch_data[b'data']
            train_labels = batch_data[b'labels']
            train_labels = to_categorical(train_labels)
            train_images = tf.reshape(train_images, [-1, 3, 32, 32])
            train_images = tf.transpose(train_images, [0, 2, 3, 1])
            train_images = train_images * 1 / 255
            i += 1
            if i == 5:
                i = 1
            yield train_images, train_labels


    def _cifar_generator(self,subset="train"):
        pattern = r".*cifar.*"
        dirs = [i for i in self.dirs if re.search(pattern, i)][0]

        if subset == "train":
            return self.__cifar_train_generator(dirs)

        elif subset == "val":
            val_data = self.unpickle(dirs + f"data_batch_{5}")
            val_images = val_data[b'data']
            val_labels = val_data[b'labels']
            val_labels = to_categorical(val_labels)
            val_images = tf.reshape(val_images, [-1, 3, 32, 32])
            val_images = tf.transpose(val_images, [0, 2, 3, 1])
            val_images = val_images*1/255
            return val_images, val_labels

        elif subset == "test":

            test_data = self.unpickle(dirs+ "test_batch")
            test_images = test_data[b'data']
            test_labels = test_data[b'labels']
            test_labels = to_categorical(test_labels)
            test_images = tf.reshape(test_images, [-1, 3, 32, 32])
            test_images = tf.transpose(test_images, [0, 2, 3, 1])
            test_images = test_images * 1 / 255
            return test_images,test_labels

        else:
            raise ValueError(f"Invalid dataset provided: {subset}. Supported datasets are 'train' and 'test'.")


    def generate_image(self,dataset,subset = "train"):
        # here we will generate image
        # might use multiple image generator as we might use more than 1 dataset
        if dataset == "tiny-imagenet":
            return self.imagenet_generator()
        elif dataset == "cifar10":
            return self._cifar_generator(subset)
        else:
            raise ValueError(f"Invalid dataset provided: {dataset}. Supported datasets are 'tiny-imagenet' and 'cifar10'.")


    @staticmethod
    def generate_adversarial_image(images,method="random",model=None, **kwargs):
        k=len(images)

        if model == None and method not in ["uniform","gaussian"]:
            raise  ValueError("fgsm and pgd requires a model")

        if method == "random":
            random.choices(["uniform","gaussian","fgsm","pgd"],k=k)
        elif method == "uniform":
            return Noise.uniform_noise(images,**kwargs)
        elif method == "gaussian":
            return Noise.gaussian_noise(images,**kwargs)
        elif method == "fgsm":
            return Noise.fgsm_noise(images,**kwargs)



