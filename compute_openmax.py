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
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import sklearn.metrics


def compute_activation_vector(model, dataloader, device, pooling=AdaptiveMaxPool2d((1, 1)), mode="train"):
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
                activation_vector = np.array(torch.cat(squeezed_latent, dim=0).cpu())
                avs[labels[i]].append(activation_vector)

    # save all the acticvation vectors as pickle file
    with open(f'avs_{mode}_maxpool.pickle', 'wb') as f:
        pickle.dump(avs, f)

    return avs


def compute_mean_activation_vector(avs, num_classes):
    # dictionary that contains the mean actication vectors for each class
    mavs = {i: None for i in range(num_classes)}

    for c in range(num_classes):
        # sum up the activation vectors (AV) for each class
        mavs[c] = np.mean(avs[c], axis=0)

    return mavs


def compute_distances(mavs, avs, num_classes):
    # dictionary that stores the eucledean distance between the activation vectors and the mean of all classes
    distances = {i: [] for i in range(num_classes)}

    # compute distance of all AV the their respective MAV
    for c, mav in mavs.items():
        for av in avs[c]:
            distances[c].append(spd.euclidean(mav, av))

    return distances


def fit_weibull_distribution(distances, tail_size, num_classes):
    # dictionary that contains the so-called "meta-recognition system" (MRS) for each class
    mrs = {i: libmr.MR() for i in range(num_classes)}

    # use the fit_high() function to fit the weibull distribution for each class
    for c, class_distances in distances.items():
        mrs[c].fit_high(
            sorted(class_distances)[-tail_size:],
            tail_size
        )

    return mrs


def compute_openmax(mrs, mavs, avs, num_classes, alpharank=10, apply_softmax=True):
    openmax_probs = []
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]

    for c, class_avs in avs.items():

        for av in class_avs:

            logits = av[:num_classes]
            openmax_known = []

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

            if apply_softmax:
                # Apply Softmax on the logits
                logits = np.exp(logits) / np.sum(np.exp(logits))

            # a vector of shape (num_classes,) with the probability of the point being an outlier
            # with respect to each class
            outlying_prob = np.asarray(w_scores) * ranked_alpha
            # modify the entry in the logits accordingly
            openmax_known = logits * (1 - outlying_prob)
            # append the outlying class probabilities as everyting that has been "taken away"
            openmax_prob = np.append(openmax_known, np.sum(logits * outlying_prob))

            if not apply_softmax:
                # apply softmax on the recalibrated logits to get the openmax probabilities
                openmax_prob = np.exp(openmax_prob) / np.sum(np.exp(openmax_prob))

            openmax_probs.append(openmax_prob)

    return openmax_probs


def calc_auroc(id_test_results, ood_test_results):
    # concat the socres
    scores = np.concatenate((id_test_results, ood_test_results))

    # this is what we would expect (see main function for explaination)
    trues = np.array(([0] * len(id_test_results)) + ([1] * len(ood_test_results)))

    # calculate AUROC
    result = roc_auc_score(trues, scores)

    return result


def calc_roc_old(scores, gts):
    pairs = [(x, y) for x, y in zip(scores, gts)]
    pairs.sort(key=lambda x: -x[0])
    fp = 0
    tp = 0
    num_p = np.sum(gts)
    prec = []
    rec = []
    fmeasure = 0
    # pdb.set_trace()
    for p in pairs:
        if p[1]:
            tp += 1
        else:
            fp += 1
        pr = float(tp) / (tp + fp)
        prec.append(pr)
        if num_p == 0:
            rc = 0
        else:
            rc = float(tp) / num_p
        rec.append(rc)
        if pr + rc != 0:
            newf = 2 * pr * rc / (pr + rc)
            fmeasure = max(newf, fmeasure)
    prs = {}
    for pr, rc in zip(prec, rec):
        if prs.get(rc) is None:
            prs[rc] = pr
        else:
            prs[rc] = max(prs[rc], pr)

    print("f =", fmeasure)
    mp = np.mean(list(prs.values()))
    print("mp =", mp)

    return fmeasure, mp


def calc_extended_accuracy(scores, gts):

    extended_accuracy = 0

    print("Extended Accuracy =", extended_accuracy)

    return extended_accuracy

def calc_tp_fu(splitList_in_dist, num_class):

    tp = 0
    fu = 0

    max_indices = []
    for prediction in splitList_in_dist:
        max_index = np.argmax(prediction)
        max_indices.append(max_index)

    for entry in max_indices:
        if entry == num_class:
            tp += 1
        elif entry == 10:
            fu += 1

    return tp, fu


def calc_tu(open_set_openmax_scores):

    tu = 0

    max_indices = []
    for prediction in open_set_openmax_scores:
        max_index = np.argmax(prediction)
        max_indices.append(max_index)

    for entry in max_indices:
        if entry == 10:
            tu += 1

    return tu

# workaround for lambda expandtion
def expand_channels(x):
    if x.shape[0] == 1:
        return x.expand(3, -1, -1)
    else:
        return x
def split_list(list, chunk_size):
    return [list[i:i+chunk_size] for i in range(0, len(list), chunk_size)]

if __name__ == "__main__":

    # Define some fixed variables
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 10
    BATCHSIZE = 1
    MEANS = [0.4914, 0.4822, 0.4465]
    STDS = [0.2023, 0.1994, 0.2010]
    TAIL_SIZE_WD = 100

    # Initialize Deep Hierarchical Network from weights
    model = DHRNet(NUM_CLASSES)  # Initialize DHR Net from pre-defined architecture
    checkpoint = torch.load("./dhr_net.pt", map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])  # Load pre-trained weights (change path if necessary)
    model.to(DEVICE)  # put model to device
    model.eval()  # set Model to eval-mode (gradients are disabled)

    # transformation applied on images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS),
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
    if os.path.isfile('./avs_train_maxpool.pickle'):
        # Load the JSON file as a dictionary
        with open('avs_train_maxpool.pickle', 'rb') as f:
            avs_train = pickle.load(f)
        print(f"Loaded Activation Vectors for each class")
    else:
        print("File 'avs_train.pickle' does not exist in the current directory. Computing...")
        # Compute the activation vectors for all images in the train dataset
        avs_train = compute_activation_vector(model, trainloader, DEVICE, mode="train")

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

    if os.path.isfile('./avs_test_maxpool.pickle'):
        with open('avs_test_maxpool.pickle', 'rb') as f:
            avs_test = pickle.load(f)
        print(f"loaded activation vectors for test data")
    else:
        print("File 'avs_test.pickle' does not exist in the current directory. Computing...")
        # compute the AV for the test images
        avs_test = compute_activation_vector(model, testloader, DEVICE, mode="test")

    # these transformation are only used for outlying datapoints and make sure they follow
    # some basic constraints such as size (32x32) and have 3 channels
    transform_outlier = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS),
        transforms.Lambda(expand_channels)  # (lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x)
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

    if os.path.isfile('./avs_outlier_maxpool.pickle'):
        with open('avs_outlier_maxpool.pickle', 'rb') as f:
            avs_outlier = pickle.load(f)
        print(f"loaded activation vectors for outlying data")
    else:
        print("File 'avs_outlier.pickle' does not exist in the current directory. Computing...")
        avs_outlier = compute_activation_vector(model, outlierloader, DEVICE, mode="outlier")

    # This function recalibrates the SoftMax scores and adds an editional unit for unkown probability and thus computes OpenMax
    # it returns the outlier probabilities (thus the additional class added by OpenMax) for each of the images

    # first compute the openmax scores for the test images (for CIPHAR-10 SHAPE=[10000, 11])
    # here we expect the last entry (outlying class score) to be close to 0, because no outliers
    in_dist_openmax_scores = compute_openmax(mrs, mavs, avs_test, NUM_CLASSES)

    # then compute the openmax scores for the outlying images (for CIPHAR-10 SHAPE=[10000, 11])
    # here we expect the last entry (outlying class score) to be close to 1, because this are outliers
    open_set_openmax_scores = compute_openmax(mrs, mavs, avs_outlier, NUM_CLASSES)

    # Create a scatterplot to plot the outlying probability for test and outlying images
    fig, ax = plt.subplots(nrows=11, ncols=1, figsize=(30, 165))
    for c in range(0, 11):
        in_dist_om_class = [om[c] for om in in_dist_openmax_scores]
        open_set_om_class = [om[c] for om in open_set_openmax_scores]
        ax[c].scatter(range(len(in_dist_om_class)), in_dist_om_class, color='blue', label='in_dist_scores')
        ax[c].scatter(range(len(in_dist_om_class), len(in_dist_om_class) + len(open_set_om_class)), open_set_om_class,
                      color='red', label='open_set_scores')
        ax[c].set_xlabel('Index', fontsize=20)
        ax[c].set_ylabel('Value', fontsize=20)
        ax[c].legend(fontsize=20)
        ax[c].set_title(f'OpenMax Scores of Class {c}', fontsize=25)
        ax[c].tick_params(axis='y', labelsize=17)
        ax[c].tick_params(axis='x', labelsize=17)
    plt.tight_layout()
    plt.savefig("openmax_scores_maxpool.png")

    # based on these assumptions, we can compute the AUROC using ONLY the outlying class score
    print("The AUROC is ",
          calc_auroc([om[-1] for om in in_dist_openmax_scores], [om[-1] for om in open_set_openmax_scores]))

    # split the list into buckets of 1000
    # With these buckets we can now calculate true positives for each class
    splitList_in_dist = split_list(in_dist_openmax_scores, 1000)
    true_postivies = 0
    #False Uknowns needed to calculate normalized Accuracy, calculated from the in_dist_openmax_scores
    false_unknowns = 0
    #True Uknowns calculated from the open_set_max_scores
    true_unknowns = 0
    tu = calc_tu(open_set_openmax_scores)
    true_unknowns += tu

    # From original paper source code
    fs = []
    mps = []
    aurocs = []
    for c in range(0, 11):
        print("class =", c)

        scores = [openmax_score[c] for openmax_score in in_dist_openmax_scores + open_set_openmax_scores]
        gts = [x == c for i in range(11) for x in ([0] * 1000 if i == 0 else ([i] * 1000 if i < 10 else [10] * 10000))]

        f, mp = calc_roc_old(scores, gts)
        if c < 10:
            tp, fu = calc_tp_fu(splitList_in_dist[c], c)
            true_postivies += tp
            false_unknowns += fu

        fs.append(f)
        mps.append(mp)

        auroc = sklearn.metrics.roc_auc_score(gts, scores)
        print("AUROC: ", auroc)
        aurocs.append(auroc)
    #Since multiclass classification accuracy can be calculated by identifying TP and dividing by observations
    accuracy = true_postivies / 10000
    #Normalized Accuracy
    normalized_accuracy = true_unknowns / (true_unknowns + false_unknowns)
    #Weight needed according to formel from paper, can be set individually
    weight = 0.5
    print("mf_known =", sum(fs[:-1]) / (len(fs) - 1))
    print("mf =", sum(fs) / len(fs))
    print("mmp =", sum(mps) / len(mps))
    print("Accuracy for 0-9 =", accuracy)
    print("Extended Accuracy =", weight * accuracy + ((1 - weight) * normalized_accuracy))
    print("Average AUROC: ", np.mean(aurocs))
