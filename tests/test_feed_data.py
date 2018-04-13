"""
Unit test class for data generator test
created in PyCharm and executable as run-debug
"""

import unittest
import numpy as np
import feed_data

class TestCreateGenerators(unittest.TestCase):
    """
    unit test class for data generator tests
    """

    def test_create_generator(self):
        """ creates generator and tests many conditions

        :return: None
        """

        testgen = feed_data.create_generators('../test_data')
        img1, msk1 = testgen.next()

        #batch size
        self.assertEqual(img1.shape[0], 8)
        #img dimensions
        self.assertEqual(img1.shape[1], 256)
        self.assertEqual(img1.shape[2], 256)
        #channel
        self.assertEqual(img1.shape[3], 1)

        #batch size
        self.assertEqual(msk1.shape[0], 8)
        #img dimensions
        self.assertEqual(msk1.shape[1], 256)
        self.assertEqual(msk1.shape[2], 256)
        #channel
        self.assertEqual(msk1.shape[3], 2)

        #one-hot on masks
        self.assertEqual(set(msk1.flatten()), {0, 1})

        self.assertEqual(img1[0].dtype, 'float32')
        self.assertEqual(msk1[0].dtype, 'uint8')

        #is zero centered
        self.assertEqual(int(np.mean(img1[0].flatten())), 0)

if __name__ == '__main__':
    unittest.main()
