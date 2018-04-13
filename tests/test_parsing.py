import unittest
import numpy as np
from .. import parsing

class TestParsing(unittest.TestCase):

    def test_parse_patient_csv(self):
        self.testcsv = '../test_data/link.csv'
        pat_lst = parsing.parse_patient_csv(self.testcsv)
        pat_id, orig_id = zip(*pat_lst)
        self.assertEqual(len(pat_id) == len(orig_id), True)

    def test_parse_contour_file(self):
        self.test_ctr_path = '../test_data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt'
        ctr = parsing.parse_contour_file(self.test_ctr_path)
        ctr_x, ctr_y = zip(*ctr)
        self.assertEqual(max(ctr_x) <= 256, True)
        self.assertEqual(max(ctr_y) <= 256, True)
        self.assertEqual(min(ctr_x) > 0, True)
        self.assertEqual(min(ctr_y) > 0, True)
        self.assertEqual(len(ctr_x), len(ctr_y))

    def test_parse_dicom_file(self):
        self.test_dicom_path = '../test_data/dicoms/SCD0000101/1.dcm'
        dcm = parsing.parse_dicom_file(self.test_dicom_path)
        self.assertEqual(type(dcm), dict)
        self.assertEqual(type(dcm['pixel_data']), np.ndarray)
        self.assertEqual(dcm['pixel_data'].shape[0] > 0, True)
        self.assertEqual(dcm['pixel_data'].shape[1] > 0, True)
        self.assertEqual(dcm['pixel_data'].shape[0] <= 256, True)
        self.assertEqual(dcm['pixel_data'].shape[1] <= 256, True)

    def test_poly_to_mask(self):
        self.test_ctr_path = '../test_data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt'
        ctr = parsing.parse_contour_file(self.test_ctr_path)
        poly = parsing.poly_to_mask(ctr, 256, 256)
        self.assertSetEqual(set(poly.flatten()), {True, False})
        self.assertEqual(poly.dtype, 'bool')

if __name__ == '__main__':
    unittest.main()
