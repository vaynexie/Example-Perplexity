'''
the aim of this code file is to use the 500 .npy files produced by 'test.py' to output the X-Perplextiy, C-Perplextiy,
Top 5 X-Confusion classes and Top C-Confusion classes for each image in ImageNet Validation set.
'''

#load needed package
import os
from tensorflow import keras
import numpy as np
import json
import pandas as pd
import tensorflow as tf
import scipy.io
from sklearn.metrics import accuracy_score
import gc
import time
from load_net_structure import load_net_structure

#Load the function 'decode_predictions' for ImageNet from tensorflow (decode the probability into probability accompanying with the label index and label name)
def load(net_name):
    _, decode_predictions = load_net_structure(net_name)
    return decode_predictions

#define function to obatin the X and C Perplexity for each image
def get_perplexity():
    # Perpare label-name-index mapping dictionaries
    current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    validation_mat = scipy.io.loadmat(current_path + "/meta.mat")["synsets"]
    validation_label = list(map(lambda x: x[0][0][0][0], validation_mat))
    validation_names = list(map(lambda x: x[0][1][0], validation_mat))
    label2name = dict(zip(validation_label, validation_names))
    name2label = dict(zip(validation_names, validation_label))
    #load the ImageNet label for the images
    with open(current_path + "ILSVRC2012_validation_ground_truth.txt") as fp:
        line = fp.read().splitlines()
        #val_gt: List of ImageNet Labels for the Images
        val_gt = np.array(list(map(lambda x: label2name[int(x)], line)))
    validation_folder = current_path + "ILSVRC2012_img_val"
    validation_image_names = np.array(os.listdir(validation_folder))
    validation_image_names.sort()
    validation_df = np.stack((validation_image_names, val_gt), axis=1)
    validation_df = pd.DataFrame(validation_df, columns=["filename", "class"])

    #Prepare to load the prediction results from .npy (record_paths-to record the paths for the .npy files)
    record_paths = []
    for path, subdirs, files in os.walk(current_path + "/test_result/"):
        for name in files:
            record_paths.append(os.path.join(path, name))
    _, decode_predictions = load("ResNet50")

    # prepare dataframe to save the X-Perplextiy and C-Perplexity
    perplexity = np.stack(
        (validation_image_names, val_gt, np.zeros(50000), np.zeros(50000)), axis=1
    )
    perplexity = pd.DataFrame(
        perplexity, columns=["filename", "label",
                             "x_perplexity", "c_perplexity"]
    )
    perplexity["x_perplexity"] = pd.to_numeric(
        perplexity["x_perplexity"], errors="coerce"
    )
    perplexity["c_perplexity"] = pd.to_numeric(
        perplexity["c_perplexity"], errors="coerce"
    )
    
 
    '''
    Prepare dataframe to save the confusion and entropy for each image in each model:
    
    We output:
    netwise_x_perplexity and netwise_c_perplexity, but they are not used in the further code
    
    output them for analyzing how the perplexity is affected by the reference classifier population
    
    for example, we have 500 models in the classifier population, we can also select 100 models according to some settings from these 500 models when needed
    
    Using netwise_x_perplexity and netwise_c_perplexity, we can output X-Perplextiy and C-Perplexity given by these 100 models easily.
    
    '''
    netwise_x_perplexity = pd.DataFrame(
        (validation_image_names), columns=["filename"])
    netwise_c_perplexity = pd.DataFrame(
        (validation_image_names), columns=["filename"])

    labels = open("imagenet_class_index.json")
    labels = json.load(labels)
    all_labels = []
    for i in range(1000):
        all_labels.append(labels[str(i)][0])

    validation_col = np.zeros((50000, 1000))

    '''
    Prepare 
    x_perplexity_confusion to record the top-1 classification results for the images
    it is a dictionary in the form of {'nxxxxxxx':50,'nxxxxxxx':50,....}
    'nxxxxxxx' is the label index and 50 means there are 50 times the image is classified as 'nxxxxxxx' among the 500 times of prediction (500 models); 
    
    c_perplexity_confusion to record the entropy value for each class for the images 
    (will produce -p*log(p) for each class of the image given one model, finally we take the average on the 500 models).
    
    
    x_perplexity_confusion, c_perplexity_confusion  will also be output, 
    but here we mainly used them for producing the Top X confusion labels and Top C confusion labels for the Image,
    just output them in case we need them in further analysis (e.g.  x_perplexity_confusion can be used to analyze the frequent confusion pairs)
    
    '''
    x_perplexity_confusion = {}
    c_perplexity_confusion = pd.DataFrame(
        validation_col, columns=all_labels, index=validation_image_names
    )
    length = len(record_paths)

    # for loop to iterate net
    for result_idx, path in enumerate(record_paths):
        start = time.time()
        record = np.load(path)
        decode = decode_predictions(record, top=1)
        
        netwise_x_perplexity[path] = np.zeros(50000)
        netwise_c_perplexity[path] = np.zeros(50000)

        # for loop to iterate each prediction result
        for step, j in enumerate(decode):
            # calculate x perplexity
            predicted_label = j[0][0]
            file_name = validation_image_names[step]
            if file_name not in x_perplexity_confusion:
                x_perplexity_confusion[file_name] = {}
            gt = val_gt[step]
            c = 0
            if predicted_label != gt:
                c += 1
            if predicted_label not in x_perplexity_confusion[file_name]:
                x_perplexity_confusion[file_name][predicted_label] = 1
            else:
                x_perplexity_confusion[file_name][predicted_label] += 1

            perplexity["x_perplexity"][step] += c
            netwise_x_perplexity[path][step] = c

        # for loop to iterate each prediction result
        for step, q in enumerate(record):
            # calculate c perplexity
            file_name = validation_image_names[step]
            class_entropy = q * np.log2(q)
            class_entropy = np.nan_to_num(class_entropy)
            entropy = -np.nansum(class_entropy, -1)
            c_perplexity_confusion.loc[file_name] += (-class_entropy) * (
                1 / length)
            perplexity["c_perplexity"][step] += entropy
            netwise_c_perplexity[path][step] = entropy

        print(result_idx, path)
        print("cost: %f" % (time.time() - start))
    perplexity["x_perplexity"] /= length
    perplexity["c_perplexity"] /= length
    perplexity["c_perplexity"] = np.power(2, perplexity["c_perplexity"])

    #do the discretization on the X-Perplexity to product the X-Perplexity class label for further analysis
    validation_X_perplexity_label = pd.cut(perplexity["x_perplexity"].astype('float'),
                                           bins=[-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                 0.6, 0.7, 0.8, 0.9, 0.9999999, 100],
                                           labels=["0", "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1", "1"])

    perplexity['x_perplexity_label'] = validation_X_perplexity_label
    #output the files
    perplexity.to_csv("perplexity.csv")
    np.save("x_perplexity_confusion", x_perplexity_confusion)
    c_perplexity_confusion.to_csv("c_perplexity_confusion.csv")
    netwise_x_perplexity.to_csv("netwise_x_perplexity.csv")
    netwise_c_perplexity.to_csv("netwise_c_perplexity.csv")

#Used to change the column names in the netwise_x_perplexity.csv and netwise_c_perplexity.csv for better use

def change_name(filename):
    '''
    Change the colomn names to net name only
    '''
    df = pd.read_csv(filename)

    coloum = df.columns
    new_coloum = []
    for i in coloum:
        if "all_validation_result" in i:
            new_name = i.split("/")[-1]
            temp = new_name.split("\\")
            new_name = r"%s\%s\\%s" % (temp[0], temp[1], temp[2])
            new_coloum.append(new_name)
        else:
            new_coloum.append(i)

    df.columns = new_coloum
    df.to_csv("%s_namechanged.csv" % filename[:-4])

#define funciton to output the top 5 X-Confusion classes and C-Confusion classes
def get_top_5_confusion_class():
    '''
    Sort the perplexity confusion value image-wise. Then get the top-5 perplexity classes
    '''
    c_perplexity_confusion = pd.read_csv("c_perplexity_confusion.csv")
    x_perplexity_confusion = np.load(
        "x_perplexity_confusion.npy", allow_pickle=True).item()
    row_idx = c_perplexity_confusion.iloc[:, 0]
    top_5_confusion = pd.read_csv("perplexity.csv")
    col_names = c_perplexity_confusion.columns[1:]

    top_5_c = []
    top_5_x = []

    for i in range(50000):
        all_confusion = c_perplexity_confusion.iloc[i, 1:]
        idx_sort = np.argsort(all_confusion)[::-1][:5]
        top_5_class_name = col_names[idx_sort].values.tolist()
        top_5_values = all_confusion[idx_sort].values.tolist()
        top_5 = list(zip(top_5_class_name, top_5_values))
        top_5_c.append(top_5)

    for j in row_idx:
        local_dic = x_perplexity_confusion[j]
        sorted_dic = sorted(local_dic.items(),
                            key=lambda x: x[1], reverse=True)
        if len(sorted_dic) > 5:
            sorted_dic = sorted_dic[:5]
        top_5_x.append(sorted_dic)
    top_5_confusion["top 5 c_perplexity class"] = top_5_c
    top_5_confusion["top 5 x_perplexity class"] = top_5_x
    top_5_confusion.to_csv("perplexity.csv")


'''
Basic code logic:

1. run get_perplexity() to get:

perplexity.csv - record the X/C perpelxtiy
x_perplexity_confusion.npy, c_perplexity_confusion.csv - record the class details for the images under the classifier population 
netwise_x_perplexity.csv, netwise_c_perplexity.csv - record the prediction details for the images under each classifier in classifier population


2. 
change_name("netwise_x_perplexity.csv")
change_name("netwise_c_perplexity.csv")

change the column names for netwise_x_perplexity.csv and netwise_c_perplexity.csv for better further use (change the column name to the model name)

3. get_top_5_confusion_class()

use the x_perplexity_confusion.npy and c_perplexity_confusion.csv to compute the top X/C confusion labels and add the information into ** perplexity.csv **.

'''
if __name__ == "__main__":
    get_perplexity()
    change_name("netwise_x_perplexity.csv")
    change_name("netwise_c_perplexity.csv")
    get_top_5_confusion_class()
