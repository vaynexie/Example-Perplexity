'''
This code is to run the 500 models on the ImageNet Validation Set (50000 images), and to produce 500 .npy files 
the .npy file includes the predicted logits for each image

'''

#load needed packages
import os
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io
from sklearn.metrics import accuracy_score
import gc
import time
from guppy import hpy
from load_net_structure import load_net_structure
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#define function to load the image pre-processing function for each structure
def load(net_name):
    preprocess_input, decode_predictions = load_net_structure(net_name)
    return preprocess_input, decode_predictions

if __name__ == "__main__":
    #load the label for the images
    current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    validation_mat = scipy.io.loadmat(current_path + "meta.mat")["synsets"]
    validation_label = list(map(lambda x: x[0][0][0][0], validation_mat))
    validation_names = list(map(lambda x: x[0][1][0], validation_mat))
    label2name = dict(zip(validation_label, validation_names))
    name2label = dict(zip(validation_names, validation_label))
    with open(current_path + "ILSVRC2012_validation_ground_truth.txt") as fp:
        line = fp.read().splitlines()
        val_gt = np.array(list(map(lambda x: label2name[int(x)], line)))
    validation_folder = current_path + "ILSVRC2012_img_val"
    validation_image_names = np.array(os.listdir(validation_folder))
    validation_image_names.sort()
    validation_df = np.stack((validation_image_names, val_gt), axis=1)
    validation_df = pd.DataFrame(validation_df, columns=["filename", "class"])
    
    #Add the path for the 500 models into nets_path (prepare for running the 500 classifiers on the dataset)
    net_name_list = [
        "ResNet50",
        "ResNet101",
        "DenseNet121",
        "DenseNet169",
        "DenseNet201",
        "EfficientNetB2",
        "EfficientNetB4",
        "Xception",
        "InceptionV3",
        "VGG16",
    ]
    dataset_name = ["1_25","2_25","3_25","1_50","2_50","3_50","1_75","2_75","3_75","100"]
    nets_path = current_path + "Nets/"

    for net_name in net_name_list:
        print(net_name)
        #set the input image size for different structures
        size = [224, 224]
        target_size_list = {
            "Xception": [299, 299],
            "EfficientNetB2": [260, 260],
            "InceptionV3": [299, 299],
        }
        if net_name in ["Xception", "EfficientNetB2", "InceptionV3"]:
            print("use user define size")
            size = target_size_list[net_name]
        net_folder_path = nets_path + net_name + "/"
        preprocess_input, decode_predictions = load(net_name)
        
        #load and pre-process the image in ImageNet validation set
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
        validation_generator = data_generator.flow_from_dataframe(
            validation_df,
            validation_folder,
            target_size=size,
            batch_size=25,
            shuffle=False,
        )
        
        #Run the model one by one on the ImageNet Validation Set
        for dataset in dataset_name:
            target_path = net_folder_path + dataset
            all_files = os.listdir(target_path)
            for j in all_files:
                if j[-2:] == "h5":
                    #load the model and make the prediction
                    net = keras.models.load_model(target_path + "/" + j)
                    prediction_result = net.predict_generator(
                        validation_generator, len(validation_generator), verbose=1
                    )
                    new_path = current_path  + "test_result/" + net_name + "/" + dataset
                    os.makedirs(new_path, exist_ok=True)
                    np.save(
                        new_path + "/" + "%s_prediction_result" % j[:-3],
                        prediction_result,
                    )
                    
                    #calculate and record the accuracy on the ImageNet validation set for each model
                    predicted_class_indices = np.argmax(prediction_result, axis=1)
                    labels = validation_generator.class_indices
                    labels = dict((v, k) for k, v in labels.items())
                    predictions = [labels[k] for k in predicted_class_indices]
                    acc_score = accuracy_score(list(val_gt), predictions)
                    accuracy = "model %s %s %s: %f\n" %(net_name, dataset, j, acc_score)
                    with open("accuracy.txt", "a") as f:
                        f.write(accuracy)
                    print(
                        "model " + str(j) + ":" + str(acc_score)
                    )
                    validation_generator.reset()
                    tf.keras.backend.clear_session()
