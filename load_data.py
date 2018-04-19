# -*- coding: utf-8 -*-
"""
Created on 12 April, 2018 @ 12:30 PM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: dicoms
License:

Description: - pipeline that parses DICOM images, contour files, creates boolean mask,
    then assigns all data to the self.data class variable
             - can access data by calling properties "labeled_dicoms" or "all_dicoms"
             - load_all_patients matches images, masks from link.csv and returns the as list
"""

import os
import sys
import numpy as np
import parsing


class PatientData(object):
    """
    Object for each patient.
    "labeled_dicoms" and "all_dicoms" properties contain image and mask information.

    self.data is a list of tuples of:
        (image_index, image, boolean inner-contour, boolean outer-contour)

    self.labeled is a list of image_index for which images have labels

    Data directory structure:
    directory = "./final_data"
        directory/contourfiles
                 /contourfiles/SC-HF-I-1
                              /i-contours
                              /o-contours
                 /contourfiles/SC-HF-I-2
                 ...
        directory/dicoms
                 /dicoms/SCD0000101
                 /dicoms/SCD0000101
                 ...
        link.csv
    """

    def __init__(self, _dicom_dir, _contour_dir, inner_outer='inner'):
        """ creating a patient instance loads all dicom images and contour files

        :param _dicom_dir: path to all patients' dicoms
        :param _contour_dir: path to all patients' contours
        :param inner_outer: 'inner', 'outer' or 'both' for contour type to load
        """

        self.data = []

        #list of indices for labeled masks
        self.labeled = []

        # iterate through all .dcm in the patient's dicom folder
        dicom_files = os.listdir(_dicom_dir)
        for dicom_file in dicom_files:
            if not dicom_file.endswith('.dcm'):
                pass
            else:
                idx = int(dicom_file[:-4])

                dcm = self.load_dicoms(_dicom_dir + "/" + dicom_file)
                height, width = dcm.shape

                ctr_i_b = None
                ctr_o_b = None
                if inner_outer == 'inner':
                    ctr_i_b = self.load_contours(_contour_dir+'/i-contours/'+
                                                 'IM-0001-{:04d}-icontour-manual.txt'.format(idx), width, height)
                elif inner_outer == 'outer':
                    ctr_o_b = self.load_contours(_contour_dir+'/o-contours/'+
                                                 'IM-0001-{:04d}-ocontour-manual.txt'.format(idx), width, height)
                elif inner_outer == 'both':
                    ctr_i_b = self.load_contours(_contour_dir+'/i-contours/'+
                                                 'IM-0001-{:04d}-icontour-manual.txt'.format(idx), width, height)
                    ctr_o_b = self.load_contours(_contour_dir+'/o-contours/'+
                                                 'IM-0001-{:04d}-ocontour-manual.txt'.format(idx), width, height)
                else:
                    ctr_i_b = None
                    ctr_o_b = None

                # if either inner or outer exists for given idx, add to self.labeled
                if ctr_i_b is not None:
                    self.labeled.append(idx)
                elif ctr_o_b is not None:
                    self.labeled.append(idx)

                self.data.append((idx, dcm, ctr_i_b, ctr_o_b))


    def load_dicoms(self, filename):
        """ loads dicoms

        :param filename: path to patient's dicom folder
        :return: pixel data as numpy array
        """
        _dcm = parsing.parse_dicom_file(filename)
        return _dcm['pixel_data']


    def load_contours(self, filename, width, height):
        """ check if contour exists, loads it, converts to binary mask
            if mask doesn't exist, returns None

        :param filename: path to patient's contour folder
        :param width: width of matching dicom image
        :param height: height of matching dicom image
        :return: binary mask, or None if no contour file exists
        """

        if os.path.isfile(filename):
            _ctr = parsing.parse_contour_file(filename)
            ctr = parsing.poly_to_mask(_ctr, width, height)
        else:
            ctr = None
        return ctr

    @property
    def labeled_dicoms(self):
        """ removes (images, mask) groups that have None for mask
            indicies in self.labeled are based on file name, not list indices (start at 1 not 0).

        :return: list of tuples: (images, i-mask, o-mask)
        """
        return [sorted(self.data)[i-1][1:] for i in self.labeled]

    @property
    def all_dicoms(self):
        """ returns all data from self.data, but removes index from first element

        :return: list of tuples: (images, i-mask, o-mask)
        """
        return [dcm_ctr_pair[1:] for dcm_ctr_pair in self.data]


def load_all_patients(data_dir, inner_outer='both', labeled_only=True):
    """  Parses link.csv to match dicoms with contours
         iterates through all patients and appends to list


    :param data_dir: folder that contains "dicoms", "contourfiles" folders and link.csv
    :param inner_outer: whether to consider 'inner', 'outer', or 'both' contour masks
    :return: list of dicom images, list of boolean inner-contours, list of boolean outer-contour
             note: None can exist in list of contours
    """

    pat_lst = parsing.parse_patient_csv(data_dir+'/link.csv')

    images_masks = []
    # first row of pat_lst is a header
    for patient in pat_lst[1:]:

        dicom_dir = data_dir +"/dicoms/"+ patient[0]
        contour_dir = data_dir +"/contourfiles/"+ patient[1]

        pat = PatientData(dicom_dir, contour_dir, inner_outer=inner_outer)
        if labeled_only:
            images_masks += pat.labeled_dicoms
        else:
            images_masks += pat.all_dicoms

    # images_masks is a list of three elements: image, inner_mask, outer_mask
    images, inner_masks, outer_masks = zip(*images_masks)

    return list(images), list(inner_masks), list(outer_masks)


def write_all_patients():
    """ if load_data called from command line, this method will
        write all images and binary masks to disk as .npy arrays

    :return: None
    """

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    imgs, i_msks, o_msks = load_all_patients(data_dir=data_dir)

    for idx, array in enumerate(imgs):
        np.save(output_dir+'/img_'+str(idx), array)
    for idx, array in enumerate(i_msks):
        np.save(output_dir+'/i_msk_'+str(idx), array)
    for idx, array in enumerate(o_msks):
        np.save(output_dir + '/o_msk_' + str(idx), array)

    return None

if __name__ == '__main__':
    write_all_patients()