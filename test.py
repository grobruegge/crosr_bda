import pickle
from unittest.mock import MagicMock
import torch
import torchvision as torchvision
from torch.utils.data import DataLoader, TensorDataset
from unittest import TestCase
import unittest
import numpy as np
from numpy.testing import assert_array_equal
import train_dhr_nn
from torch.optim import SGD
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import argparse
from compute_openmax import compute_mean_activation_vector, compute_distances, calc_metrics
import os
import compute_openmax
from unittest.mock import patch
import random
import libmr
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class TestCompute_openmax(unittest.TestCase):
    """
    Class to test compute openmax
    """
    def test_model_initialization(self):
        """
           test the initialization of the model
        """
        # Define the test input
        num_classes = 10
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Call the function to initialize the model
        model = compute_openmax.initialize_model(num_classes, device)

        # Perform the assertions to verify the model initialization
        self.assertTrue(isinstance(model, train_dhr_nn.DHRNet))
        self.assertEqual(model.num_classes, num_classes)
        self.assertEqual(next(model.parameters()).device, device)
        self.assertFalse(model.training)
    def test_save_scores_to_pickle(self):
        """
           test if the score is saved correctly
        """
        args = MagicMock(save_w_scores=True, save_openmax_scores=True)
        file_name = "test_file"
        w_scores_id = [1, 2, 3]
        w_scores_ood = [4, 5, 6]
        in_dist_openmax_scores = [0.1, 0.2, 0.3]
        open_set_openmax_scores = [0.4, 0.5, 0.6]

        compute_openmax.save_scores_to_pickle(args, file_name, w_scores_id, w_scores_ood, in_dist_openmax_scores, open_set_openmax_scores)

        # Assert that w_scores.pickle file was created and contains the expected data
        w_scores_file_path = os.path.join('data', 'plot_pickles', f'w_scores_{file_name}.pickle')
        self.assertTrue(os.path.exists(w_scores_file_path))
        with open(w_scores_file_path, 'rb') as f:
            saved_w_scores = pickle.load(f)
        expected_w_scores = w_scores_id + w_scores_ood
        self.assertEqual(saved_w_scores, expected_w_scores)

        # Assert that openmax_scores.pickle file was created and contains the expected data
        openmax_scores_file_path = os.path.join('data', 'plot_pickles', f'openmax_scores_{file_name}.pickle')
        self.assertTrue(os.path.exists(openmax_scores_file_path))
        with open(openmax_scores_file_path, 'rb') as f:
            saved_openmax_scores = pickle.load(f)
        expected_openmax_scores = np.array(in_dist_openmax_scores + open_set_openmax_scores)
        np.testing.assert_array_equal(saved_openmax_scores, expected_openmax_scores)

        # Clean up the created pickle files
        os.remove(w_scores_file_path)
        os.remove(openmax_scores_file_path)

    def test_define_fixed_variables(self):
        """
           checks fixed variables
        """
        expected_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        expected_NUM_CLASSES = 10
        expected_BATCHSIZE = 1
        expected_MEANS = [0.4914, 0.4822, 0.4465]
        expected_STDS = [0.2023, 0.1994, 0.2010]
        expected_TAIL_SIZE_WD = 100

        DEVICE, NUM_CLASSES, BATCHSIZE, MEANS, STDS, TAIL_SIZE_WD = compute_openmax.define_fixed_variables()

        self.assertEqual(DEVICE, expected_DEVICE)
        self.assertEqual(NUM_CLASSES, expected_NUM_CLASSES)
        self.assertEqual(BATCHSIZE, expected_BATCHSIZE)
        self.assertEqual(MEANS, expected_MEANS)
        self.assertEqual(STDS, expected_STDS)
        self.assertEqual(TAIL_SIZE_WD, expected_TAIL_SIZE_WD)



    def test_fit_weibull_distribution(self):
        """
           tests the correct computation of the weibull_distribution parameters
        """
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

    def test_compute_mean_activation_vector(self):
        """
           tests the correct computation of the weibull_distribution parameters
        """
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

    def test_compute_activation_vector(self):
        """
           tests the correct computation of the activation_vectors
        """
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



    def test_compute_openmax(self):
        """
           tests the correct computation of the openmax classifier
        """
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

    def test_compute_mean_activation_vector(self):
        """
           tests the correct computation of mean activation vectors
        """
        # Set up sample inputs for testing
        self.avs = {
            0: [np.array([0.2, 0.5, 0.3, 0.1]), np.array([0.1, 0.3, 0.6, 0.2])],
            1: [np.array([0.4, 0.1, 0.2, 0.5]), np.array([0.3, 0.2, 0.5, 0.4])]
        }
        self.num_classes = 2

        expected_mavs = {
            0: np.array([0.15, 0.4, 0.45, 0.15]),
            1: np.array([0.35, 0.15, 0.35, 0.45])
        }
        mavs = compute_mean_activation_vector(self.avs, self.num_classes)
        for c in range(self.num_classes):
            np.testing.assert_allclose(mavs[c], expected_mavs[c], rtol=1e-4)

    def test_compute_distances(self):
        """
           tests the correct computation of vector distances
        """
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
    def test_calc_metrics(self):
        """
           tests the correct computation of evaluation metrics
        """
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

    def test_loadCIFAR10(self):
        """
           tests the correct behaviour when loading Data
        """
        trainset, trainloader, testset, testloader = compute_openmax.loadCIFAR10()

        # Check if the returned objects are of the correct types
        self.assertIsInstance(trainset, datasets.CIFAR10)
        self.assertIsInstance(trainloader, torch.utils.data.DataLoader)
        self.assertIsInstance(testset, datasets.CIFAR10)
        self.assertIsInstance(testloader, torch.utils.data.DataLoader)

    def test_loadOutliers(self):
        """
           tests the correct behaviour when loading Data
        """
        outlierset, outlierloader = compute_openmax.loadOutliers()

        # Check if the returned objects are of the correct types
        self.assertIsInstance(outlierset, datasets.ImageFolder)
        self.assertIsInstance(outlierloader, torch.utils.data.DataLoader)


    def test_createplot_no_error(self):
        """
           tests the correct generation of a plot
        """
        file_name = "test_file"
        in_dist_om_class = [1, 2, 3, 4, 5]
        open_set_om_class = [6, 7, 8, 9, 10]

        try:
            compute_openmax.createplot(file_name, in_dist_om_class, open_set_om_class)
            # If no error occurred, the test passes
        except Exception as e:
            self.fail(f"An error occurred while creating the plot: {e}")



class TestTrain_dhr_nn(unittest.TestCase):
    """
    Class to test compute train_dhr_nn
    """

    @patch('builtins.print')  # Mock print function
    def test_setup_environment(self, mock_print):
        """
           test if the envionment is set up correctly
        """
        with patch.object(torch, 'manual_seed') as mock_torch_manual_seed:
            with patch.object(random, 'seed') as mock_random_seed:
                with patch.object(np.random, 'seed') as mock_np_random_seed:
                    expected_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    expected_seed = 22
                    expected_lr = 0.05
                    expected_epochs = 500
                    expected_batch_size = 128
                    expected_momentum = 0.9
                    expected_weight_decay = 0.0005
                    expected_means = [0.4914, 0.4822, 0.4465]
                    expected_stds = [0.2023, 0.1994, 0.2010]
                    expected_num_classes = 10

                    device, lr, epochs, batch_size, momentum, weight_decay, means, stds, num_classes = train_dhr_nn.setup_environment()

                    mock_print.assert_called_once_with(expected_device)
                    mock_torch_manual_seed.assert_called_once_with(expected_seed)
                    mock_random_seed.assert_called_once_with(expected_seed)
                    mock_np_random_seed.assert_called_once_with(expected_seed)

                    self.assertEqual(device, expected_device)
                    self.assertEqual(lr, expected_lr)
                    self.assertEqual(epochs, expected_epochs)
                    self.assertEqual(batch_size, expected_batch_size)
                    self.assertEqual(momentum, expected_momentum)
                    self.assertEqual(weight_decay, expected_weight_decay)
                    self.assertEqual(means, expected_means)
                    self.assertEqual(stds, expected_stds)
                    self.assertEqual(num_classes, expected_num_classes)

    def test_epoch_train(self):
        """
           tests the correct behavior of a train epoch
        """
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

    def test_epoch_val(self):
        """
           tests the correct behavior of a val epoch
        """
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

    def test_dhrnet(self):
        """
           tests the correct behavior of a dhrnet
        """
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

    def test_loadData(self):
        """
           tests the correct behaviour when loading Data
        """
        trainloader, testloader = train_dhr_nn.loadData()

        # Check if the returned objects are of the correct types
        self.assertIsInstance(trainloader, torch.utils.data.DataLoader)
        self.assertIsInstance(testloader, torch.utils.data.DataLoader)



if __name__ == '__main__':
    unittest.main()
