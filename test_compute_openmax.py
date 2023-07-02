import unittest
import numpy as np
from torch._jit_internal import ignore

import compute_openmax
from compute_openmax import compute_mean_activation_vector, calc_tu, calc_tp_fu, calculate_acc_extAcc, compute_distances
class TestCalcRocOld(unittest.TestCase):
    def test_calc_roc_old(self):
        scores = [0.5, 0.5, 0.5, 0.5] # Example scores list
        gts = [1, 0, 1, 0] # Example ground truth list

        fmeasure, mp = compute_openmax.calc_roc_old(scores, gts)

        # Assert the expected values
        self.assertAlmostEqual(fmeasure, 0.8)
        self.assertAlmostEqual(mp, 0.83, places=2)

    def test_auroc(self):
        id_test_results = np.array([0.8, 0.6, 0.4, 0.2])
        ood_test_results = np.array([0.9, 0.7, 0.5, 0.1])
        expected_result = 0.5625  # Since ood_test_results have higher scores than id_test_results

        result = compute_openmax.calc_auroc(id_test_results, ood_test_results)

        self.assertAlmostEqual(result, expected_result, places=5)

class ComputeMeanActivationVectorTest(unittest.TestCase):

    def setUp(self):
        # Set up sample inputs for testing
        self.avs = {
            0: [np.array([0.2, 0.5, 0.3, 0.1]), np.array([0.1, 0.3, 0.6, 0.2])],
            1: [np.array([0.4, 0.1, 0.2, 0.5]), np.array([0.3, 0.2, 0.5, 0.4])]
        }
        self.num_classes = 2

    def test_compute_mean_activation_vector(self):
        expected_mavs = {
            0: np.array([0.15, 0.4, 0.45, 0.15]),
            1: np.array([0.35, 0.15, 0.35, 0.45])
        }
        mavs = compute_mean_activation_vector(self.avs, self.num_classes)
        for c in range(self.num_classes):
            np.testing.assert_allclose(mavs[c], expected_mavs[c], rtol=1e-4)


class ComputeDistancesTest(unittest.TestCase):

    def test_compute_distances(self):
        mavs = {
            0: [1.0, 2.0, 3.0],
            1: [4.0, 5.0, 6.0]
        }
        avs = {
            0: [
                [0.5, 1.5, 2.5],
                [1.5, 2.5, 3.5]
            ],
            1: [
                [3.5, 4.5, 5.5],
                [4.5, 5.5, 6.5]
            ]
        }
        num_classes = 2

        expected_distances = {
            0: [0.8660254037844386, 0.8660254037844386],
            1: [0.8660254037844386, 0.8660254037844386]
        }

        distances = compute_distances(mavs, avs, num_classes)

        for c in range(num_classes):
            self.assertAlmostEqual(distances[c], expected_distances[c], places=6)

class CalcTpFuTest(unittest.TestCase):

    def test_calc_tp_fu(self):
        splitList_in_dist = [
            [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
        ]
        num_class = 3

        tp, fu = calc_tp_fu(splitList_in_dist, num_class)

        self.assertEqual(tp, 1)
        self.assertEqual(fu, 1)

class CalcTuTest(unittest.TestCase):

    def test_calc_tu(self):
        open_set_openmax_scores = [
            [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
        ]

        tu = calc_tu(open_set_openmax_scores)

        self.assertEqual(tu, 1)

@unittest.skip("Unsufficient amount of Data")
class CalculateAccExtAccTest(unittest.TestCase):

    def test_calculate_acc_extAcc(self):
        in_dist_openmax_scores = [
            [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
        ]
        open_set_openmax_scores = [
            [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
        ]

        accuracy, normalized_accuracy = calculate_acc_extAcc(in_dist_openmax_scores, open_set_openmax_scores)

        self.assertAlmostEqual(accuracy, 0.0002, delta=1e-5)
        self.assertAlmostEqual(normalized_accuracy, 0.5, delta=1e-5)

if __name__ == '__main__':
    unittest.main()