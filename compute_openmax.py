import torch
from torch.nn import AdaptiveAvgPool2d
from train_dhr_nn import DHRNet
from torchvision import datasets, transforms
import libmr
import numpy as np
import scipy.spatial.distance as spd
from tqdm import tqdm
import pickle
import os
from sklearn.metrics import roc_auc_score

def compute_activation_vector(model, dataloader, device):


    # initialize a dictionary to store the activation vectors (AV) for each class
    avs = {i: [] for i in range(model.num_classes)}

    # Iterate through batches of the dataset
    for (inputs, labels) in tqdm(dataloader):
        
        inputs, labels = inputs.to(device), np.array(labels.cpu())

        with torch.no_grad():
            # run inference of model (caclulates the model output for the whole batch)
            logits, _, latent_layers = model(inputs)

        # determine predicted class based on the max value of the logits (of site num_classes)
        predicted_classes = torch.argmax(logits, dim=1)

        # iterate through each image the batch 
        for i, predicted_class in enumerate(predicted_classes):

            # Only use the Avtivation Vectors of correctly classified samples
            if predicted_class == labels[i]:
                
                squeezed_latent = []
                # append the logits to the AV
                squeezed_latent.append(logits[i])

                # append the 3 latent representation to the AV
                for latent_layer in latent_layers[i]:
                    avg_pool = AdaptiveAvgPool2d((1,1))
                    latent_repr = torch.squeeze(avg_pool(latent_layer))
                    squeezed_latent.append(latent_repr)

                # contains the logits and the 3 latent layer representations concatinated in one vector
                activation_vector = np.array(torch.cat(squeezed_latent,dim=0).cpu())

                avs[labels[i]].append(activation_vector)

    # save all the acticvation vectors as pickle file
    with open('avs.pickle', 'wb') as f:
        pickle.dump(avs, f)

    return avs

def compute_mean_activation_vector(avs):

    # dictionary that contains the mean actication vectors for each class
    mavs = {i: None for i in range(len(avs))} 

    # sum up the activation vectors (AV) for each class
    for c, class_avs in avs.items():
        if len(class_avs) > 0:
            mavs[c] = np.mean(class_avs, axis=0)
    
    # divide through the total amount of AV for each class to get the mean
    for c in range(model.num_classes):
        mavs[c] /= len(avs[c]) 
    
    # save the MAV as file
    with open('mavs.pickle', 'wb') as f:
        pickle.dump(mavs, f)

    return mavs

def compute_distances(mavs, avs):
    # dictionary that stores the eucledean distance between the activation vectors and the mean of all classes
    distances = {i: [] for i in range(len(mavs))} 

    # compute distance of all AV the their respective MAV
    for c, mav in mavs.items():
        for av in avs[c]:
            distances[c].append(spd.euclidean(mav, av))

    # save the distances as file
    with open('distances.pickle', 'wb') as f:
        pickle.dump(mavs, f)

    return distances

def fit_weibull_distribution(distances, tail_size):

    # dictionary that contains the so-called "meta-recognition system" (MRS) for each class
    mrs = {i: libmr.MR() for i in range(len(distances))} 

    # use the fit_high() function to fit the weibull distribution for each class
    for c, class_distances in distances.items():
        mrs[c].fit_high(
            sorted(class_distances)[-tail_size:],
            tail_size
        )

    # save the MRS for each class as file 
    with open('mrs.pickle', 'wb') as f:
        pickle.dump(mrs, f)

    return mrs

def compute_openmax(mrs, mavs, avs, alpharank=10):
    
    results = []

    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]


    for c, class_avs in avs.items():
        
        for av in class_avs:
            
            openmax_known = []
            openmax_unknown = 0

            ranked_list = av.argsort().ravel()

            ranked_alpha = np.zeros(av.shape[0])
            for i in range(len(alpha_weights)):
                ranked_alpha[ranked_list[i]] = alpha_weights[i]

            for idx in range(len(av)):
                
                # get the w-score by inserting the distance of the AV and the MAV into the 
                # Weibull distribution of the respective class
                w_score = mrs[c].w_score(spd.euclidean(mavs[c], av))
                # modify the entry in the AV accordingly
                modified_unit = av[idx] * ( 1 - w_score*ranked_alpha[idx])
                openmax_known.append(modified_unit)
                openmax_unknown += (av[idx] - modified_unit)

                # calculate the probability of this AV being unknown
                prob_unknowns = np.exp(openmax_unknown)/(np.sum(np.exp(np.asarray(openmax_known))) + np.exp(openmax_unknown))

            results.append(prob_unknowns)

    return results

def calc_auroc(id_test_results, ood_test_results):
    
    # concat the socres
    scores = np.concatenate((id_test_results, ood_test_results))
    
    # this is what we would expect (see main function for explaination)
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))

    # calculate AUROC
    result = roc_auc_score(trues, scores)

    return result   
   
if __name__ == "__main__":

    # Define some fixed variables
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 10
    BATCHSIZE = 1
    MEANS = [0.4914, 0.4822, 0.4465]
    STDS = [0.2023, 0.1994, 0.2010]
    TAIL_SIZE_WD = 20

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
    if os.path.isfile('./mavs.pickle') and os.path.isfile('./avs.pickle'):
        # Load the JSON file as a dictionary
        with open('mavs.pickle', 'rb') as f:
            mavs = pickle.load(f)
        with open('avs.pickle', 'rb') as f:
            avs_train = pickle.load(f)
        print(f"Loaded (Mean) Activation Vectors for each class")
    else:
        print("File 'mavs.pickle' and/or 'avs_train.pickle' does not exist in the current directory. Computing...")
        # Compute the activation vectors for all images in the train dataset
        avs_train = compute_activation_vector(model, trainloader, DEVICE)
        # Compute the mean activation vector for all classes
        mavs = compute_mean_activation_vector(avs_train)

    if os.path.isfile('./distances.pickle'):
        with open('distances.pickle', 'rb') as f:
            distances = pickle.load(f)
        print(f"Loaded Distances to Mean Activation Vectors for each class")
    else: 
        print("File 'distances.pickle' does not exist in the current directory. Computing...")
        # Compute distance of all AV the their respective MAV
        distances = compute_distances(mavs, avs_train)

    if os.path.isfile('./mrs.pickle'):
        with open('mrs.pickle', 'rb') as f:
            mrs = pickle.load(f)
        print(f"Loaded Weibull Distribution (Meta Recognition System) for each class")
    else: 
        print("File 'mrs.pickle' does not exist in the current directory. Computing...")
        # fit the weibull distribution (called meta-recognition system, MRS) for every class
        mrs = fit_weibull_distribution(distances, TAIL_SIZE_WD)

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

    # compute the AV for the test images
    avs_test = compute_activation_vector(model, testloader, DEVICE)

    # these transformation are only used for outlying datapoints and make sure they follow 
    # some basic constraints such as size (32x32) and have 3 channels
    transform_outlier = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(MEANS,STDS),
        transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x)
    ])
    
    # load IMAGENET as outlying dataset from folder
    # download the dataset from "" and put it into ./data/Imagenet 
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

    if os.path.isfile('./avs_outlier.pickle'):
        with open('avs_outlier.pickle', 'rb') as f:
            avs_outlier = pickle.load(f)
        print(f"loaded activation vectors for outlying data")
    else: 
        print("File 'avs_outlier.pickle' does not exist in the current directory. Computing...")
        avs_outlier = compute_activation_vector(model, outlierloader, DEVICE, mode="outlier")

    # This function recalibrates the SoftMax scores and adds an editional unit for unkown probability and thus computes OpenMax 
    # it returns the outlier probabilities (thus the additional class added by OpenMax) for each of the images

    # first compute for the test images (here we expect the scores to be close to 0, because no outliers)
    in_dist_scores = compute_openmax(mrs, mavs, avs_test)
    # then compute for the outlying images (here we expect the scores to be close to 1, because no outliers)
    open_set_scores = compute_openmax(mrs, mavs, avs_outlier)

    # based on these assumptions, we can compute the AUROC
    print("The AUROC is ",calc_auroc(in_dist_scores, open_set_scores))