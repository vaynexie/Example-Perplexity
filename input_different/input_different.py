'''
In order to allow more feasible testing on our image perplexity estimation method, 
we design this code to allow user to use their classifier populaiton on different datasets apart from ImageNet.
'''

#load needed package
import os
import numpy as np
import json
import pandas as pd
import tensorflow as tf
import scipy.io
import gc
import time


#function for reading the top1 prediction from the logits
def decode_predictions(preds, label_dict, top=1):
  results=[]
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    i=top_indices[0]
    result =[label_dict[str(top_indices[0])][0],pred[top_indices[0]]] 
    results.append(result)
  return results

#function for calculating the x/c perplexity
def get_perplexity(current_path,predict_result_path,ground_truth_df,label_dict):
    predict_result_list=os.listdir(predict_result_path)
    image_names_list=list(ground_truth_df['filename'])
    ground_truth_list=list(ground_truth_df['class'])
    record = np.load(predict_result_path+'/'+predict_result_list[0])
    num_image=record.shape[0]
    num_class=record.shape[1]
    perplexity = np.stack(
        (image_names_list, ground_truth_list, np.zeros(num_image), np.zeros(num_image)), axis=1)
    perplexity = pd.DataFrame(
        perplexity, columns=["filename", "label",
                             "x_perplexity", "c_perplexity"])
    perplexity["x_perplexity"] = pd.to_numeric(
        perplexity["x_perplexity"], errors="coerce")
    perplexity["c_perplexity"] = pd.to_numeric(
        perplexity["c_perplexity"], errors="coerce")

    #prepare dataframe to save the confusion and entropy for each image in each model
    netwise_x_perplexity = pd.DataFrame(
        (image_names_list), columns=["filename"])
    netwise_c_perplexity = pd.DataFrame(
        (image_names_list), columns=["filename"]) 
    all_labels = []
    for i in range(num_class):
        all_labels.append(label_dict[str(i)][0])

    validation_col = np.zeros((num_image, num_class))
    # prepare perplexity for each class
    x_perplexity_confusion = {}
    c_perplexity_confusion = pd.DataFrame(
        validation_col, columns=all_labels, index=image_names_list
    )

    length = len(predict_result_list)

    # for loop to iterate net
    for path in predict_result_list:
        start = time.time()
        record = np.load(predict_result_path+'/'+path)
        decode = decode_predictions(record, label_dict, top=1)
        netwise_x_perplexity[path] = np.zeros(num_image)
        netwise_c_perplexity[path] = np.zeros(num_image)
        # for loop to iterate each prediction result
        for step, j in enumerate(decode):
            # calculate x perplexity
            predicted_label = j[0]
            #print(predicted_label)
            file_name = image_names_list[step]
            if file_name not in x_perplexity_confusion:
                x_perplexity_confusion[file_name] = {}
            gt = ground_truth_list[step]
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
            file_name = image_names_list[step]
            class_entropy = q * np.log2(q)
            class_entropy = np.nan_to_num(class_entropy)
            entropy = -np.nansum(class_entropy, -1)
            c_perplexity_confusion.loc[file_name] += (-class_entropy) * (
                1 / length)
            perplexity["c_perplexity"][step] += entropy
            netwise_c_perplexity[path][step] = entropy
        print(path)
        print("cost: %f" % (time.time() - start))
    perplexity["x_perplexity"] /= length
    perplexity["c_perplexity"] /= length
    perplexity["c_perplexity"] = np.power(2, perplexity["c_perplexity"])
       
    #do the discretization on the X-Perplexity to product the X-Perplexity class label for further analysis
    X_perplexity_label = pd.cut(perplexity["x_perplexity"].astype('float'),
                                           bins=[-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                 0.6, 0.7, 0.8, 0.9, 0.9999999, 100],
                                           labels=["0", "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1", "1"])

    perplexity['x_perplexity_label'] = X_perplexity_label
    #output the files
    perplexity.to_csv("perplexity.csv")
    np.save("x_perplexity_confusion", x_perplexity_confusion)
    c_perplexity_confusion.to_csv("c_perplexity_confusion.csv")
    netwise_x_perplexity.to_csv("netwise_x_perplexity.csv")
    netwise_c_perplexity.to_csv("netwise_c_perplexity.csv")


#define funciton to output the top 5 X-Confusion classes and C-Confusion classes
def get_top_5_confusion_class(current_path):
    '''
    Sort the perplexity confusion value image-wise. Then get the top-5 perplexity classes
    '''
    c_perplexity_confusion = pd.read_csv(current_path+"c_perplexity_confusion.csv")
    num_image=c_perplexity_confusion.shape[0]
    x_perplexity_confusion = np.load(
       current_path+ "x_perplexity_confusion.npy", allow_pickle=True).item()
    row_idx = c_perplexity_confusion.iloc[:, 0]
    top_5_confusion = pd.read_csv(current_path+"perplexity.csv")
    col_names = c_perplexity_confusion.columns
    top_5_c = []
    top_5_x = []
    for i in range(num_image):
        all_confusion = c_perplexity_confusion.iloc[i, 1:]
        idx_sort = np.argsort(all_confusion)[::-1][:5]
        top_5_class_name = col_names[idx_sort].values.tolist()
        top_5_values = all_confusion[idx_sort].values.tolist()
        top_5 = list(zip(top_5_class_name, top_5_values))
        top_5_c.append(top_5)
    for j in row_idx:
        local_dic = x_perplexity_confusion[j]
        sorted_dic = sorted(local_dic.items(),key=lambda x: x[1], reverse=True)
        if len(sorted_dic) > 5:
            sorted_dic = sorted_dic[:5]
        top_5_x.append(sorted_dic)
    top_5_confusion["top 5 c_perplexity class"] = top_5_c
    top_5_confusion["top 5 x_perplexity class"] = top_5_x
    top_5_confusion.to_csv(current_path+"perplexity.csv")
    


'''
input
1. folder predict_result that contains .npy files that record the array of predicted logits for each image under each classifier in classifier population.

2. label_dict.json:
include a dictionary that records the order of classes in the output layer of the classifier

3. result_df.pkl:

include a dataframe that records the ground truth label for the dataset
'''

if __name__ == "__main__":
	current_path=os.getcwd()
	current_path=current_path+'/'
	predict_result_path='predict_result'
	ground_truth_df = pd.read_pickle("result_df.pkl")
	label_dict= open("label_dict.json")
	label_dict = json.load(label_dict)
	get_perplexity(current_path,predict_result_path,ground_truth_df,label_dict)
	get_top_5_confusion_class(current_path)

