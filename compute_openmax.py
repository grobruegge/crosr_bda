import torch
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from train_dhr_nn import DHRNet
from torchvision import datasets, transforms
import libmr
import numpy as np
import scipy.spatial.distance as spd
from tqdm import tqdm
import pickle
import os
from sklearn import metrics 
from matplotlib import pyplot as plt
import argparse
from tabulate import tabulate

def compute_activation_vector(args, model, dataloader, device, pooling=AdaptiveMaxPool2d((1,1)), mode="train"):
    """
    Computes activation vectors for a model based on the input data from the dataloader.

    Args:
        args (object): A collection of arguments or settings for the function.
        model (torch.nn.Module): The model for which activation vectors are to be computed.
        dataloader (torch.utils.data.DataLoader): A dataloader object providing the input data for the model.
        device (torch.device): The device on which the computations should be performed (e.g., 'cuda' for GPU or 'cpu' for CPU).
        pooling (torch.nn.Module, optional): A pooling module applied to the latent layers to form the activation vectors. By default, `AdaptiveMaxPool2d((1,1))` is used.
        mode (str, optional): The mode in which the function is executed. By default, it is set to "train".

    Returns:
        dict: A dictionary containing the activation vectors for each class.

    """

    # initialize a dictionary to store the activation vectors (AV) for each class
    avs = {i: [] for i in range(model.num_classes)}
    correct_predictions = 0
    current_total = 0

    # initialize tqdm progress bar for dynamic printing
    pbar = tqdm(total=len(dataloader), desc="")

    # Iterate through batches of the dataset
    for (inputs, labels) in dataloader:
        
        inputs, labels = inputs.to(device), np.array(labels.cpu())

        with torch.no_grad():
            # run inference of model (caclulates the model output for the whole batch)
            logits, _, latent_layers = model(inputs)

        # determine predicted class based on the max value of the logits (of site num_classes)
        predicted_classes = torch.argmax(logits, dim=1)

        # used to calculate the accuracy
        correct_predictions += (predicted_classes.numpy() == labels).sum()
        current_total += labels.shape[0]
        # print acc to console
        tqdm.set_description(pbar, f"Current Accuracy: {correct_predictions / current_total * 100:.2f}% ")
        pbar.update(labels.shape[0])

        # iterate through each image the batch 
        for i, predicted_class in enumerate(predicted_classes):

            # Only use the Avtivation Vectors of correctly classified samples
            if predicted_class == labels[i] or mode != "train":
                
                squeezed_latent = []
                # append the logits to the AV
                squeezed_latent.append(logits[i])

                # append the 3 latent representation to the AV
                for latent_layer in latent_layers:
                    latent_repr = torch.squeeze(pooling(latent_layer))
                    squeezed_latent.append(latent_repr)

                # contains the logits and the 3 latent layer representations concatinated in one vector
                activation_vector = np.array(torch.cat(squeezed_latent,dim=0).cpu())
                avs[labels[i]].append(activation_vector)

    # save all the acticvation vectors as pickle file
    with open(os.path.join('data', 'features', f'avs_{mode}_{args.feature_suffix}.pickle'), 'wb') as f:
        pickle.dump(avs, f)

    return avs

def compute_mean_activation_vector(avs, num_classes):
    """
    Computes the mean activation vectors for each class based on the provided activation vectors.

    Args:
        avs (dict): A dictionary containing the activation vectors for each class.
        num_classes (int): The total number of classes.

    Returns:
        dict: A dictionary containing the mean activation vectors for each class.

    """

    # dictionary that contains the mean actication vectors for each class
    mavs = {i: None for i in range(num_classes)} 

    for c in range(num_classes):
        # sum up the activation vectors (AV) for each class
        mavs[c] = np.mean(avs[c], axis=0) 

    return mavs

def compute_distances(mavs, avs, num_classes):
    """
    Computes the Euclidean distances between the activation vectors and the mean activation vectors for each class.

    Args:
        mavs (dict): A dictionary containing the mean activation vectors for each class.
        avs (dict): A dictionary containing the activation vectors for each class.
        num_classes (int): The total number of classes.

    Returns:
        dict: A dictionary containing the Euclidean distances between the activation vectors and the mean activation vectors for each class.

    """

    # dictionary that stores the eucledean distance between the activation vectors and the mean of all classes
    distances = {i: [] for i in range(num_classes)} 
    
    # compute distance of all AV the their respective MAV
    for c, mav in mavs.items():
        # distances[c] = np.linalg.norm(avs[c] - mav, axis=1)
        for av in avs[c]:
            distances[c].append(spd.euclidean(mav, av))
         
    return distances

def fit_weibull_distribution(distances, tail_size, num_classes):
    """
    Fits the Weibull distribution to the given distances for each class to create a "meta-recognition system" (MRS).

    Args:
       distances (dict): A dictionary containing the Euclidean distances between the activation vectors and the mean activation vectors for each class.
       tail_size (int): The size of the tail used for fitting the Weibull distribution.
       num_classes (int): The total number of classes.

    Returns:
       dict: A dictionary containing the fitted Weibull distribution (meta-recognition system) for each class.

    """

    # dictionary that contains the so-called "meta-recognition system" (MRS) for each class
    mrs = {i: libmr.MR() for i in range(num_classes)} 

    # use the fit_high() function to fit the weibull distribution for each class
    for c, class_distances in distances.items():

        mrs[c].fit_high(
            sorted(class_distances)[-tail_size:],
            tail_size
        )

    return mrs

def compute_openmax(mrs, mavs, avs, num_classes, apply_softmax=True, alpharank=10):
    """
    Computes the OpenMax probabilities and the w-scores for each sample based on the given meta-recognition system (MRS), mean activation vectors (MAVs), and activation vectors (AVs).

    Args:
        mrs (dict): A dictionary containing the fitted Weibull distribution (meta-recognition system) for each class.
        mavs (dict): A dictionary containing the mean activation vectors for each class.
        avs (dict): A dictionary containing the activation vectors for each class.
        num_classes (int): The total number of classes.
        apply_softmax (bool, optional): Flag indicating whether to apply softmax on the logits before computing OpenMax probabilities. By default, it is set to True.
        alpharank (int, optional): The rank used to calculate the alpha weights for the OpenMax scores. By default, it is set to 10.

    Returns:
        tuple: A tuple containing the OpenMax probabilities and the w-scores for each sample.

    """
    
    w_scores_and_logits = []
    openmax_probs = []
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]

    for c, class_avs in avs.items():
        
        for av in class_avs:

            logits = av[:num_classes]

            ranked_list = np.argsort(av[:num_classes])[::-1]
            ranked_alpha = np.zeros(num_classes)
            for i in range(alpharank):
                ranked_alpha[ranked_list[i]] = alpha_weights[i]

            # There are two option to calculate the OpenMax Scores:
            # 1) Use the logits to multiply with w-scores: This has the disadvantage that when calculating 
            # the outlying class score, you sum over pos. and neg. values which may cancel out
            # 2) Use the Softmax of logits to multiply with w-scores: It seems that this has been adapted by the
            # paper CROSR and IMHO it makes more sense
            w_scores = []
            for idx in range(num_classes):
                # get the w-score by inserting the distance of the AV and the MAV into the 
                # Weibull distribution of the respective class
                w_score = mrs[idx].w_score(spd.euclidean(mavs[idx], av))
                w_scores.append(w_score)

            w_scores_and_logits.append((np.stack(w_scores), logits))

            if apply_softmax:
                # Apply Softmax on the logits
                logits = np.exp(logits) / np.sum(np.exp(logits))

            # a vector of shape (num_classes,) with the probability of the point being an outlier
            # with respect to each class
            outlying_prob = np.asarray(w_scores) * ranked_alpha

            # modify the entry in the logits accordingly
            openmax_known = logits * (1 - outlying_prob)
            # append the outlying class probabilities as everyting that has been "taken away"
            openmax_prob = np.append(openmax_known, np.sum(np.maximum(logits * outlying_prob, 0)))
            # openmax_prob = np.append(openmax_known, np.sum(logits * outlying_prob, 0))

            if not apply_softmax:
                # apply softmax on the recalibrated logits to get the openmax probabilities 
                openmax_prob = np.exp(openmax_prob) / np.sum(np.exp(openmax_prob))

            openmax_probs.append(openmax_prob)

    return openmax_probs, w_scores_and_logits

def calc_metrics(in_dist_openmax_scores, open_set_openmax_scores):
    """
    Calculates various evaluation metrics including False Positive Rates (FPR), True Positive Rates (TPR),
    thresholds, Area Under the Receiver Operating Characteristic curve (AUROC), Youden's J-statistic, optimal
    cut-off thresholds, F1-Score, and Accuracy for each class.

    Args:
        in_dist_openmax_scores (list): A list of in-distribution OpenMax scores for each class.
        open_set_openmax_scores (list): A list of open-set OpenMax scores for each class.

    Returns:
        str: A formatted table displaying the evaluation metrics for each class.

    """
    
    fpr = dict() # False-Positive-Rates (FPR) for each class
    tpr = dict() # True Positive-Rates (TPR) for each class
    thresholds = dict() # Thresholds for each class
    roc_auc = dict() # AUROC for each class
    j_stats = dict() #Youden's J-statistic for each class
    opt_thresholds = dict() # Optimal Cut-Off Threshold based on Youden's J-statistic for each class
    f1_scores = dict() # F1-Score using argmax for each class
    f1_scores_w_cutoff = dict() # F1-Score using Cut-Off Threshold for each class
    acc_scores = dict() # Accuracy Scores for each class

    # iterate through all possible classes
    for c in range(NUM_CLASSES + 1):
        
        # Only look at the scores of this class iteration
        scores = [om[c] for om in in_dist_openmax_scores + open_set_openmax_scores]
        # Create a list of lenght 20'000 containing only True for the class indices of this loop iteration
        trues = [x == c for i in range(NUM_CLASSES + 1) for x in ([0] * 1000 if i == 0 else ([i] * 1000 if i < 10 else [10] * 10000))]

        # receive the FPR, TPR and choosen thresholds
        fpr[c], tpr[c], thresholds[c] = metrics.roc_curve(trues, scores, drop_intermediate=False)

        # calculate AUROC score
        roc_auc[c] = metrics.auc(fpr[c], tpr[c])

        # calculate optimal threshold based on Youden's J-statistic
        j_stats[c] = tpr[c] - fpr[c]
        opt_thresholds[c] = thresholds[c][np.argmax(j_stats[c])]
        
        argmax_predictions = np.argmax(in_dist_openmax_scores + open_set_openmax_scores, axis=-1) == c
        threshold_predictions = [1 if score >= opt_thresholds[c] else 0 for score in scores]

        # calculate F1 score using argmax
        f1_scores[c] = metrics.f1_score(trues, argmax_predictions)
        # calculate F1-Score using cutoff-index
        f1_scores_w_cutoff[c] = metrics.f1_score(trues, threshold_predictions)

        acc_scores[c] = metrics.accuracy_score(trues, argmax_predictions)


    # create a table to be printed in the console
    column_headers = ["Class", "Accuracy", "F1-Score using argmax", "ROC-AUC", "Optimal Cut-Off Threshold", "F1-Score using Cut-Off"]
    row_headers = list(range(11))

    table = tabulate(
        tabular_data=[[acc_scores.get(c, None), f1_scores.get(c, None), roc_auc.get(c, None), opt_thresholds.get(c, None), f1_scores_w_cutoff.get(c, None)] for c in range(NUM_CLASSES + 1)],
        headers=column_headers, 
        showindex=row_headers, 
        floatfmt=".4f"
    )

    return table
    
if __name__ == "__main__":

    # Define some fixed variables
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 10
    BATCHSIZE = 1
    MEANS = [0.4914, 0.4822, 0.4465]
    STDS = [0.2023, 0.1994, 0.2010]
    TAIL_SIZE_WD = 100

    parser = argparse.ArgumentParser(description='Get activation vectors')
    # We can compute the features in different ways, e.g., using MaxPooling, AvgPooling or not considering the latent repr.
    parser.add_argument('--feature_suffix',default='avgpool',type=str)
    # We can either compute the SoftMax BEFORE the OpenMax calculation (as done in this paper) or AFTER the OpenMax calculation
    # as done in the original paper of OpenMax
    parser.add_argument('--apply_softmax_before',action='store_true')
    parser.add_argument('--save_openmax_scores',action='store_true')
    parser.add_argument('--save_w_scores',action='store_true')
    args = parser.parse_args()

    # Initialize Deep Hierarchical Network from weights
    model = DHRNet(NUM_CLASSES) # Initialize DHR Net from pre-defined architecture
    checkpoint = torch.load("./dhr_net.pt", map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict']) # Load pre-trained weights (change path if necessary)
    model.to(DEVICE) # put model to device
    model.eval() # set Model to eval-mode (gradients are disabled)

    # transformation applied on images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEANS,STDS),
    ])

    # load CIFAR-10 train dataset and create a dataloader
    trainset = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=BATCHSIZE,
        shuffle=False, 
        num_workers=2
    )

    # Check if the JSON file exists in the current directory
    if os.path.isfile(os.path.join('data', 'features', f'avs_train_{args.feature_suffix}.pickle')):
        # Load the JSON file as a dictionary
        with open(os.path.join('data', 'features', f'avs_train_{args.feature_suffix}.pickle'), 'rb') as f:
            avs_train = pickle.load(f)
        print(f"Loaded Activation Vectors for each class")
    else:
        print("File 'avs_train.pickle' does not exist in the current directory. Computing...")
        # Compute the activation vectors for all images in the train dataset
        avs_train = compute_activation_vector(args, model, trainloader, DEVICE, mode="train")
    
    # Compute the mean activation vector for all classes
    mavs = compute_mean_activation_vector(avs_train, NUM_CLASSES)

    # Compute distance of all AV the their respective MAV
    distances = compute_distances(mavs, avs_train, NUM_CLASSES)

    # fit the weibull distribution (called meta-recognition system, MRS) for every class
    mrs = fit_weibull_distribution(distances, TAIL_SIZE_WD, NUM_CLASSES)

    # load CIFAR-10 test dataset and create dataloader
    testset = datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=BATCHSIZE,
        shuffle=False, 
        num_workers=2
    )

    if os.path.isfile(os.path.join('data', 'features', f'avs_test_{args.feature_suffix}.pickle')):
        with open(os.path.join('data', 'features', f'avs_test_{args.feature_suffix}.pickle'), 'rb') as f:
            avs_test = pickle.load(f)
        print(f"loaded activation vectors for test data")
    else: 
        print("File 'avs_test.pickle' does not exist in the current directory. Computing...")
        # compute the AV for the test images
        avs_test = compute_activation_vector(args, model, testloader, DEVICE, mode="test")

    # these transformation are only used for outlying datapoints and make sure they follow 
    # some basic constraints such as size (32x32) and have 3 channels
    transform_outlier = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(MEANS,STDS),
        transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x)
    ])
    
    # load IMAGENET as outlying dataset from folder
    # download the dataset from "https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz" and put it into ./data/Imagenet 
    outlierset = datasets.ImageFolder(
        root=os.path.join('data', 'Imagenet'), 
        transform=transform_outlier
    )
    outlierloader = torch.utils.data.DataLoader(
        outlierset, 
        batch_size=BATCHSIZE,
        shuffle=False, 
        num_workers=2
    )

    if os.path.isfile(os.path.join('data', 'features', f'avs_outlier_{args.feature_suffix}.pickle')):
        with open(os.path.join('data', 'features', f'avs_outlier_{args.feature_suffix}.pickle'), 'rb') as f:
            avs_outlier = pickle.load(f)
        print(f"loaded activation vectors for outlying data")
    else: 
        print("File 'avs_outlier.pickle' does not exist in the current directory. Computing...")
        avs_outlier = compute_activation_vector(args, model, outlierloader, DEVICE, mode="outlier")

    # This function recalibrates the SoftMax scores and adds an editional unit for unkown probability and thus computes OpenMax 
    # it returns the outlier probabilities (thus the additional class added by OpenMax) for each of the images

    # first compute the openmax scores for the test images (for CIPHAR-10 SHAPE=[10000, 11])
    # here we expect the last entry (outlying class score) to be close to 0, because no outliers
    in_dist_openmax_scores, w_scores_id = compute_openmax(mrs, mavs, avs_test, NUM_CLASSES, apply_softmax=args.apply_softmax_before)

    # then compute the openmax scores for the outlying images (for CIPHAR-10 SHAPE=[10000, 11])
    # here we expect the last entry (outlying class score) to be close to 1, because this are outliers
    open_set_openmax_scores, w_scores_ood = compute_openmax(mrs, mavs, avs_outlier, NUM_CLASSES, apply_softmax=args.apply_softmax_before)
    
    # file name to save plots and scores
    file_name = f'{args.feature_suffix}_{TAIL_SIZE_WD}_{"softmax_before" if args.apply_softmax_before else "softmax_after"}'

    if args.save_w_scores:
        with open(os.path.join('data', 'plot_pickles', f'w_scores_{file_name}.pickle'), 'wb') as f:
            pickle.dump(w_scores_id+w_scores_ood, f)

    if args.save_openmax_scores:
        with open(os.path.join('data', 'plot_pickles', f'openmax_scores_{file_name}.pickle'), 'wb') as f:
            pickle.dump(np.array(in_dist_openmax_scores+open_set_openmax_scores), f)

    # filter out the outlying class probabilit
    in_dist_om_class = [om[-1] for om in in_dist_openmax_scores]
    open_set_om_class = [om[-1] for om in open_set_openmax_scores]

    # Create a scatterplot to plot the outlying probability for test and outlying images
    fig, ax = plt.subplots(figsize=(30, 15))
    ax.scatter(range(len(in_dist_om_class)), in_dist_om_class, color='blue', label='in_dist_scores')
    ax.scatter(range(len(in_dist_om_class), len(in_dist_om_class)+len(open_set_om_class)), open_set_om_class, color='red', label='open_set_scores')
    ax.set_xlabel('Index', fontsize=20)
    ax.set_ylabel('Value', fontsize=20)
    ax.tick_params(axis='y', labelsize=17)
    ax.tick_params(axis='x', labelsize=17)
    ax.set_title(f'OpenMax Scores of outlying class', fontsize=25)
    ax.legend(fontsize=20)
    plt.tight_layout()

    plt.savefig(os.path.join('data', 'plots', f'openmax_scores_{file_name}.png'))

    table = calc_metrics(in_dist_openmax_scores, open_set_openmax_scores)

    print(table)