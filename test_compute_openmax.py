import unittest

import numpy as np

import compute_openmax

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

if __name__ == '__main__':
    unittest.main()