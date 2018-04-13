# -*- coding: utf-8 -*-
"""
Created on 12 April, 2018 @ 12:40 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: dicoms
License:

Description:  generator endpoint of data pipeline towards training a model

"""

import load_data
import numpy as np

from keras import utils
from keras.preprocessing.image import ImageDataGenerator


def create_generators(datafolder, batch_size=8, shuffle=True, normalize=True):
    """

    :param datafolder: path to folder with dicom and contourfile folders
    :param batch_size: number of images in each batch outputted by generator
    :param shuffle: boolean to shuffle the data before output
    :param normalize: boolean to mean-zero center all images
    :return: an iterator yielding tuples of (images, targets), where both are numpy arrays
    """

    images, masks = load_data.load_all_patients(data_dir=datafolder)
    images = np.asarray(images)
    masks = np.asarray(masks)

    # channels last
    images = images[..., np.newaxis]

    # usually no standard deviation normalization for images
    images = images.astype('float64')
    if normalize:
        images -= np.mean(images, axis=(1, 2), keepdims=True)

    #one-hot encode masks
    try:
        dims = masks.shape
        classes = len(set(masks.flatten()))  # get num classes from first image
        assert classes == 2, 'number mask train classes not equal 2'
        new_shape = dims + (classes,)
        masks = utils.to_categorical(masks).reshape(new_shape)
    except AssertionError:
        print('assertion error, num classes not equal 2')
    masks = masks.astype('uint8')

    #use Keras ImageDataGenerator module
    idg = ImageDataGenerator()

    train_generator = idg.flow(images, masks,
                               batch_size=batch_size, shuffle=shuffle)

    return train_generator