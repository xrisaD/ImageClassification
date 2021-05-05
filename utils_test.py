import unittest
import numpy as np
import numpy.testing as nt

from datasets import Dataset

import utils


class MyTestCase(unittest.TestCase):
    def test_softmax_1(self):
        x = np.array([[-1, 0, 3, 5], [-1, 0, 3, 5]])
        res = utils.softmax(x)
        expected = [[0.0021657, 0.00588697, 0.11824302, 0.87370431], [0.0021657, 0.00588697, 0.11824302, 0.87370431]]
        self.assertEqual(len(res), len(expected))
        nt.assert_array_almost_equal(res, expected)

    def test_softmax_2(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]])
        res = utils.softmax(x)
        expected = [[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813],
                    [0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813]]
        self.assertEqual(len(res), len(expected))
        nt.assert_array_almost_equal(res, expected)

    def test_dataset1(self):
        d = Dataset("mnist/test")
        p = iter(d)
        x1, y1 = next(p)
        x2, y2 = next(p)
        eq = (x1 == x2).all()
        self.assertFalse(eq)
        self.assertEqual(y1.shape[0], x1.shape[0])


if __name__ == '__main__':
    unittest.main()
