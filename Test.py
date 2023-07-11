import torch
import torchvision as torchvision
from torch.utils.data import DataLoader, TensorDataset
from unittest import TestCase
import unittest
import numpy as np
from numpy.testing import assert_array_equal
import compute_openmax
import train_dhr_nn
from torch.optim import SGD
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import argparse
from compute_openmax import compute_mean_activation_vector, compute_distances, calc_metrics
import os
import libmr
from torch.utils.data import Dataset



class TestFitWeibullDistribution(unittest.TestCase):
    def test_fit_weibull_distribution(self):
        # Example distances for class 0 and 1
        distances = {
            0: [1, 2, 3, 4, 5],
            1: [2, 4, 6, 8, 10],
        }
        tail_size = 3
        num_classes = 2

        # calculate mrs and libmr.MR() verify parameter
        mrs = compute_openmax.fit_weibull_distribution(distances, tail_size, num_classes)

        para = mrs[1].get_params()
        expected_para = (3.387857964985435, 1.9212484258227571, 1, 1, 6.0)

        # Assert the expected behavior
        self.assertIsInstance(mrs, dict)
        self.assertEqual(para,expected_para)
        self.assertEqual(len(mrs), num_classes)
        for c in range(num_classes):
            self.assertIn(c, mrs)
class TestComputeMeanActivationVector(unittest.TestCase):
    def test_compute_mean_activation_vector(self):
        # Define sample activation vectors for two classes
        avs = {
            0: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            1: np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
        }

        num_classes = 2

        expected_result = {
            0: np.array([4., 5., 6.]),
            1: np.array([13., 14., 15.])
        }


        # Call the function to compute mean activation vectors
        result = compute_openmax.compute_mean_activation_vector(avs, num_classes)

        # Compare the expected result with the actual result
        for key in result:
            assert_array_equal(result[key], expected_result[key])


class TestComputeActivationVector(unittest.TestCase):

    def setUp(self):
        # Call model for testing
        self.model = train_dhr_nn.DHRNet(10)

        # Create some dummy data
        num_samples = 100
        input_shape = (3, 32, 32)
        num_classes = 10
        inputs = torch.randn(num_samples, *input_shape)
        labels = torch.randint(0, num_classes, (num_samples,))

        # Create a DataLoader from the dummy data
        dataset = TensorDataset(inputs, labels)
        self.dataloader = DataLoader(dataset, batch_size=1)

    def test_compute_activation_vector(self):
        # Set the device to CPU
        device = torch.device("cpu")

        # Set up parser
        parser = argparse.ArgumentParser(description='Get activation vectors')
        parser.add_argument('--feature_suffix', default='avgpool', type=str)
        args = parser.parse_args([])

        # Call the function under test
        avs = compute_openmax.compute_activation_vector(args, self.model, self.dataloader, device, mode="train")

        # Check if accuracy calculation is correct
        total_samples = len(self.dataloader.dataset)
        expected_accuracy = 100.0  # All predictions are correct
        computed_accuracy = sum(len(activation_vectors) for activation_vectors in avs.values()) / total_samples * 100
        self.assertIsNot(computed_accuracy, expected_accuracy)


        # delete File so that compute openmax works properly
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Specify the relative path to the file
        relative_path = 'data/features/avs_train_avgpool.pickle'

        # Construct the file path using os.path.join()
        file_path = os.path.join(current_dir, relative_path)

        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
            print(f"The file {file_path} has been deleted.")

class Testtest_epoch_train(TestCase):
    def test_epoch_train(self):
        # Set up the full dataset
        full_dataset = CIFAR10(root='data/', train=True, download=True, transform=ToTensor())

        # Define the number of samples to include in the subset
        subset_size = 1000

        # Create a subset of the dataset with a smaller number of samples
        subset_indices = torch.randperm(len(full_dataset))[:subset_size]
        subset_dataset = Subset(full_dataset, subset_indices)

        trainloader = DataLoader(subset_dataset, batch_size=32, shuffle=True)

        # Set up the network and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = train_dhr_nn.DHRNet(10)
        net.to(device)
        optimizer = SGD(net.parameters(), lr=0.01)

        # Call the epoch_train function
        accuracy, cls_loss, reconst_loss, total_loss = train_dhr_nn.epoch_train(net, device, trainloader, optimizer)

        #sencond Test for if Clause to cover 6 more statements

        # Perform assertions to check the results
        assert accuracy >= 0.0 and accuracy <= 100.0, "Accuracy should be between 0 and 100"
        assert cls_loss >= 0.0, "Classification loss should be non-negative"
        assert reconst_loss >= 0.0, "Reconstruction loss should be non-negative"
        assert total_loss >= 0.0, "Total loss should be non-negative"

        print("Unit test passed.")

class EpochValTest(unittest.TestCase):
    def test_epoch_val(self):
        # Define the test data
        batch_size = 128
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2023, 0.1994, 0.2010]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size= batch_size,
            shuffle=False,
            num_workers=2
        )
        num_classes = 10
        net = train_dhr_nn.DHRNet(num_classes)
        net.to(device)

        # Call the function to be tested
        result = train_dhr_nn.epoch_val(net, device, testloader)

        # Perform assertions on the result
        self.assertEqual(len(result), 4)  # Check if the result contains 4 values
        self.assertIsInstance(result[0], float)  # Check if the accuracy is a float
        self.assertIsInstance(result[1], float)  # Check if the classification loss is a float
        self.assertIsInstance(result[2], float)  # Check if the reconstruction loss is a float
        self.assertIsInstance(result[3], float)  # Check if the total loss is a float



class DHRNetTest(unittest.TestCase):
    def test_dhrnet(self):
        # Define the test data
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_classes = 10
        net = train_dhr_nn.DHRNet(num_classes)
        net.to(device)

        # Create a random input tensor
        input_tensor = torch.randn(1, 3, 32, 32).to(device)

        # Call the forward method of the network
        logits, reconstruct, _ = net(input_tensor)

        # Perform assertions on the output
        self.assertEqual(logits.shape, (1, num_classes))  # Check the shape of the logits
        self.assertEqual(reconstruct.shape, (1, 3, 32, 32))  # Check the shape of the reconstructed tensor


class TestComputeOpenmax(unittest.TestCase):
    def test_compute_openmax(self):
        #calulate mrs with example distances
        # Example distances for class 0 and 1
        distances = {
            0: [1, 2, 3, 4, 5],
            1: [2, 4, 6, 8, 10],
        }
        tail_size = 3
        num_classes = 2

        mrs = compute_openmax.fit_weibull_distribution(distances, tail_size, num_classes)
       #example mavs and avs
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
        alpharank = 10
        apply_softmax = True

        #Calculate openmax
        openmax_probs = compute_openmax.compute_openmax(mrs, mavs, avs, num_classes, alpharank, apply_softmax)


        # Assert the expected behavior
        self.assertEqual(len(openmax_probs), len(avs))



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


class CalcMetricsTestCase(unittest.TestCase):
    def test_calc_metrics(self):
        # Mock input data
        num_entries = 20000

        in_dist_openmax_scores = []
        open_set_openmax_scores = []

        for i in range(num_entries):
            scores = np.array([0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4])  # Adjust the scores as desired
            in_dist_openmax_scores.append(scores)
            open_set_openmax_scores.append(scores)

        in_dist_openmax_scores = np.array(in_dist_openmax_scores)
        open_set_openmax_scores = np.array(open_set_openmax_scores)

        # Mock expected output
        expected_table = """
  Class    Accuracy    F1-Score using argmax    ROC-AUC    Optimal Cut-Off Threshold    F1-Score using Cut-Off
-------  ----------  -----------------------  ---------  ---------------------------  ------------------------
      0      0.9500                   0.0000     0.5000                       1.2000                    0.0000
      1      0.9500                   0.0000     0.5000                       1.4000                    0.0000
      2      0.0500                   0.0952     0.5000                       2.4000                    0.0000
      3      0.9500                   0.0000     0.5000                       1.0000                    0.0000
      4      0.9500                   0.0000     0.5000                       1.0000                    0.0000
      5      0.9500                   0.0000     0.5000                       1.0000                    0.0000
      6      0.9500                   0.0000     0.5000                       1.0000                    0.0000
      7      0.9500                   0.0000     0.5000                       1.0000                    0.0000
      8      0.9500                   0.0000     0.5000                       1.0000                    0.0000
      9      0.9500                   0.0000     0.5000                       1.6000                    0.0000
     10      0.5000                   0.0000     0.5000                       1.8000                    0.0000
        """

        # Call the function
        table = calc_metrics(in_dist_openmax_scores, open_set_openmax_scores)

        # Assert the result
        self.assertMultiLineEqual(table.strip(), expected_table.strip())


if __name__ == '__main__':
    unittest.main()
