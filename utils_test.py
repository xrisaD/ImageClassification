import unittest
import numpy as np
import numpy.testing as nt

import utils


class MyTestCase(unittest.TestCase):
        def softmax_test(self):
            x = np.array([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]])
            res = utils.softmax(x)
            expected = [[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813], [0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813]]
            self.assertEqual(len(res), len(expected))
            nt.assert_array_equal(res, expected)

if __name__ == '__main__':
    unittest.main()
