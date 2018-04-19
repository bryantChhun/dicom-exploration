import unittest
import numpy as np
import load_data

class TestLoadData(unittest.TestCase):

    def test_PatientData(self):
        dicom_path = "../test_data/dicoms/SCD0000101"
        contour_path = "../test_data/contourfiles/SC-HF-I-1"

        p = load_data.PatientData(_dicom_dir=dicom_path, _contour_dir=contour_path,  inner_outer = 'inner')
        self.assertEqual(len(p.data[0]), 4)
        self.assertEqual(type(p.data[0][0]), int)
        self.assertEqual(type(p.data[0][1]), np.ndarray)
        self.assertEqual(type(p.labeled), list)
        self.assertEqual(type(p.labeled[0]), int)

        dcm = p.load_dicoms(filename='../test_data/dicoms/SCD0000101/48.dcm')
        self.assertEqual(dcm.dtype, 'int16')
        self.assertEqual(type(dcm), np.ndarray)
        self.assertEqual(dcm.shape[0] > 0, True)
        self.assertEqual(dcm.shape[1] > 0, True)
        self.assertEqual(dcm.shape[0] <= 256, True)
        self.assertEqual(dcm.shape[1] <= 256, True)

        ctr = p.load_contours(filename='../test_data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt',
                              width=256, height=256)
        self.assertSetEqual(set(ctr.flatten()), {True, False})
        self.assertEqual(ctr.dtype, 'bool')
        self.assertEqual(type(ctr), np.ndarray)
        self.assertEqual(ctr.shape[0], 256)
        self.assertEqual(ctr.shape[1], 256)

        self.assertEqual(len(p.all_dicoms), 240)
        self.assertEqual(len(p.labeled_dicoms), 18)
        self.assertEqual(len(p.all_dicoms[0]), 3)
        self.assertEqual(len(p.labeled_dicoms[0]), 3)


    def test_load_all_patients(self):
        imgs, i_msks, o_msks = load_data.load_all_patients(data_dir='../test_data')
        self.assertEqual(len(imgs), len(i_msks))
        self.assertEqual(len(imgs), len(o_msks))
        self.assertEqual(len(imgs), 36)
        self.assertEqual(len(i_msks), 36)
        self.assertEqual(len(o_msks), 36)
        self.assertEqual(type(imgs[0]), np.ndarray)
        self.assertEqual(imgs[0].dtype, 'int16')
        self.assertEqual(type(next(item for item in i_msks if item is not None)), np.ndarray)
        self.assertEqual(type(next(item for item in o_msks if item is not None)), np.ndarray)
        self.assertEqual(next(item for item in i_msks if item is not None).dtype, 'bool')
        self.assertEqual(next(item for item in o_msks if item is not None).dtype, 'bool')


if __name__ == '__main__':
    unittest.main()