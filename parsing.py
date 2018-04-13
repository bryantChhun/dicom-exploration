"""Parsing code for DICOMS and contour files"""

import csv
import dicom
from dicom.errors import InvalidDicomError

import numpy as np
from PIL import Image, ImageDraw


def parse_patient_csv(filename):
    """Parse the given csv filename to match contour and dicom data

    :param filename: filepath to .csv data
    :return: list of tuples with folder names (string) for dicom, contours (dicom, contour)
    """

    patient_lst = []
    try:
        with open(filename, 'r') as csvfile:
            patient_csv = csv.reader(csvfile, delimiter=',')
            for row in patient_csv:
                patient_lst.append((row[0], row[1]))
    except Exception as ex:
        print(ex)
        return None

    return patient_lst


def parse_contour_file(filename):
    """Parse the given contour filename

    :param filename: filepath to the contourfile to parse
    :return: list of tuples holding x, y coordinates of the contour
    """

    coords_lst = []
    try:
        with open(filename, 'r') as infile:
            for line in infile:
                coords = line.strip().split()

                x_coord = float(coords[0])
                y_coord = float(coords[1])
                coords_lst.append((x_coord, y_coord))
    except Exception as ex:
        print(ex)
        return None

    return coords_lst


def parse_dicom_file(filename):
    """Parse the given DICOM filename

    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        dcm_dict = {'pixel_data' : dcm_image}
        return dcm_dict
    except InvalidDicomError:
        return None


def poly_to_mask(polygon, width, height):
    """Convert polygon to mask

    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    try:
        img = Image.new(mode='L', size=(width, height), color=0)
        ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
        mask = np.array(img).astype(bool)
    except Exception as ex:
        print(ex)
        return None

    return mask
