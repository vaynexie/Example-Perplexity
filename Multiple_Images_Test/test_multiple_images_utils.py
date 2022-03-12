
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from load_net_structure import load_net_structure
import time
import matplotlib.pyplot as plt
import re
import gc
from sklearn.metrics import accuracy_score
import scipy.io
import json
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import multiprocessing as mp
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # do not use GPU boost
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def load(net_name):
    preprocess_input, decode_predictions = load_net_structure(net_name)
    return preprocess_input, decode_predictions

# Commented out IPython magic to ensure Python compatibility.


def test(file_name, net_name_list, test_mode):
    # without GPU
    # 260s one image
    image_file_name = file_name
    current_path = os.getcwd() + "/"

    validation_mat = scipy.io.loadmat(current_path + "/meta.mat")["synsets"]
    validation_label = list(map(lambda x: x[0][0][0][0], validation_mat))
    validation_names = list(map(lambda x: x[0][1][0], validation_mat))
    validation_realworld_name = list(map(lambda x: x[0][2][0], validation_mat))
    validation_gloss = list(map(lambda x: x[0][3][0], validation_mat))

    label2name = dict(zip(validation_label, validation_names))
    name2label = dict(zip(validation_names, validation_label))

    names2realworld_name = dict(
        zip(validation_names, validation_realworld_name))
    names2gloss = dict(zip(validation_names, validation_gloss))

    with open(current_path + "ILSVRC2012_validation_ground_truth.txt") as fp:
        line = fp.read().splitlines()
        val_gt = np.array(list(map(lambda x: label2name[int(x)], line)))

    nets_path = current_path + "Nets/"
    number_of_img = len(image_file_name)

    for net_name in net_name_list:
        print(net_name)
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
        image_batch_tensor = np.zeros((number_of_img, size[0], size[1], 3))

        for number, img in enumerate(image_file_name):
            img_raw = tf.io.read_file(img)
            img_tensor = tf.image.decode_image(img_raw)
            img_tensor = tf.image.resize(img_tensor, size)
            img_tensor = preprocess_input(img_tensor)
            image_batch_tensor[number] = img_tensor

        record_paths = []
        all_subdirs = []
        for path, subdirs, files in os.walk(nets_path + net_name + "/"):
            for subdir in subdirs:
                all_subdirs.append(subdir)

        for subdir in all_subdirs:
            temp_file_list = []
            for path, subdirs, files in os.walk(nets_path + net_name + "/" + subdir):
                for name in files:
                    temp_file_list.append(os.path.join(path, name))
            if test_mode == 1:
                last_digits = list(map(lambda x: x[-10:-2], temp_file_list))
                last_digits = list(
                    map(lambda x: re.sub('[^0-9]', '', x), last_digits))
                last_digits = np.array(last_digits, dtype=int)
                largest_index = np.argmax(last_digits)
                record_paths.append(temp_file_list[largest_index])
            elif test_mode == 2:
                record_paths.append(temp_file_list)
            else:
                raise ValueError
        record_paths = np.reshape(np.array(record_paths), -1)

        for single_path in record_paths:

            start_time = time.time()
            net = keras.models.load_model(single_path)

            result_path = single_path.replace(
                "Nets", "BatchTestResult")

            result_path = result_path.replace("h5", "npy")
            prediction_result = net(image_batch_tensor)

            result_folder_path = result_path.split("model")[0]

            os.makedirs(result_folder_path, exist_ok=True)

            np.save(
                result_path,
                prediction_result,
            )

            prediction_record = (
                "model: %s_%s"
                % (
                    net_name,
                    single_path,
                )
            )

            print(prediction_record)
            tf.keras.backend.clear_session()
            print("prediction cost %f second" %
                  (time.time() - start_time))


def multi_thread_optimize_test(file_name, net_name_list, test_mode, threads=6):

    pool = mp.Pool(processes=threads)  # 定义CPU核数量为5
    for i in net_name_list:
        pool.apply_async(test, args=(file_name, i, test_mode))

    pool.close()
    pool.join()
    # test(file_name, ["ResNet50"], test_mode)


def get_perplexity(file_name, img_gt, test_mode, how_many_top=5):
    current_path = os.path.dirname(os.path.realpath(__file__)) + "/"

    # get all nets' prediction results
    validation_mat = scipy.io.loadmat(current_path + "/meta.mat")["synsets"]
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

    record_paths = []
    for path, subdirs, files in os.walk(current_path + "BatchTestResult/"):
        for name in files:
            record_paths.append(os.path.join(path, name))
    _, decode_predictions = load("ResNet50")

    labels = open("imagenet_class_index.json")
    labels = json.load(labels)

    all_labels = []
    all_describtions = []
    for i in range(1000):
        all_labels.append(labels[str(i)][0])
        all_describtions.append(labels[str(i)][1])

    labels2describ = dict(zip(all_labels, all_describtions))

    length = len(record_paths)
    number_of_batch = len(file_name)

    # for loop to iterate net
    x_perplexity_confusion = []
    c_perplexity_confusion = []
    x_perplexity = []
    c_perplexity = []
    netwise_x_perplexity = []
    netwise_c_perplexity = []

    for i in range(number_of_batch):
        x_perplexity_confusion.append({})
        c_perplexity_confusion.append(np.zeros(1000))
        x_perplexity.append(0)
        c_perplexity.append(0)
        netwise_x_perplexity.append(np.zeros(length))
        netwise_c_perplexity.append(np.zeros(length))

    for net_index, path in enumerate(record_paths):
        # start = time.time()
        record = np.load(path)
        decode = decode_predictions(record, top=1)
        # for loop to iterate each prediction result
        for step, j in enumerate(decode):
            # calculate x perplexity

            predicted_label = j[0][0]
            # print(predicted_label, img_gt)

            if predicted_label != img_gt[step]:
                x_perplexity[step] += 1
                netwise_x_perplexity[step][net_index] = 1
            if predicted_label not in x_perplexity_confusion[step]:
                x_perplexity_confusion[step][predicted_label] = 1/length
            else:
                x_perplexity_confusion[step][predicted_label] += 1/length

        for step, q in enumerate(record):
            class_entropy = q * np.log2(q)
            class_entropy = np.nan_to_num(class_entropy)
            entropy = -np.nansum(class_entropy, -1)
            c_perplexity_confusion[step] += (-class_entropy)
            c_perplexity[step] += entropy
            netwise_c_perplexity[step][net_index] = entropy

    print(length, "networks are used")
    x_perplexity /= np.array(length)
    c_perplexity /= np.array(length)
    c_perplexity = np.power(2, c_perplexity)
    c_perplexity_confusion = list(
        map(lambda x: x/length, c_perplexity_confusion))

    all_labels = np.array(all_labels)
    all_describtions = np.array(all_describtions)

    top_5_c = []
    top_5_x = []

    for i in range(number_of_batch):
        idx_sort = np.argsort(c_perplexity_confusion)[i][::-1][:how_many_top]
        top_5_c_class_name = all_labels[idx_sort]
        top_5_c_describ_name = all_describtions[idx_sort]
        top_5_c_values = c_perplexity_confusion[i][idx_sort]
        top_5_c_values = np.around(top_5_c_values, 3)
        top_5_c.append(list(
            zip(top_5_c_class_name, top_5_c_describ_name, top_5_c_values)))

        sorted_dic = sorted(x_perplexity_confusion[i].items(),
                            key=lambda x: x[1], reverse=True)
        if len(sorted_dic) > how_many_top:
            sorted_dic = sorted_dic[:how_many_top]
        top_5_x_single = []
        for i in sorted_dic:
            top_5_x_single.append(
                (i[0], labels2describ[i[0]], np.round(i[1], 3)))
        top_5_x.append(top_5_x_single)

    true_labels = list(map(lambda x: labels2describ[x], img_gt))

    if test_mode != 1:
        print("true label is", img_gt, true_labels)
        print("x perplexity is", x_perplexity)
    print("c perplexity is", c_perplexity)
    for i, item in enumerate(top_5_c):
        print("for image %d, top %d c confusion classes are" %
              (i+1, how_many_top), item)
    if test_mode != 1:
        for i, item in enumerate(top_5_x):
            print("for image %d, top %d x confusion classes are" %
                  (i+1, how_many_top), item)

    save_result_foder_path = current_path + "BatchTestRecord/"
    os.makedirs(save_result_foder_path, exist_ok=True)

    np.save(save_result_foder_path + "top_x", top_5_x)
    if test_mode != 1:
        np.save(save_result_foder_path + "top_c", top_5_c)
        np.save(save_result_foder_path + "c_perplexity", c_perplexity)
    np.save(save_result_foder_path + "x_perplexity", x_perplexity)

    return top_5_c, top_5_x, x_perplexity, c_perplexity


def get_example_images(all_img_name, top_5, x_or_c):
    current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    validation_folder = current_path + "ILSVRC2012_img_val/"
    val_perplexity = pd.read_csv("perplexity.csv")

    for step, img_name in enumerate(all_img_name):
        img = img_name.split(".")[0]
        folder_path = current_path + "BatchTestSampleImages/" + img + "/" + \
            "confusion_class_samples/" + x_or_c + "/"
        os.makedirs(folder_path, exist_ok=True)
        top_5_label = np.array(top_5[step])[:, 0]
        top_5_label_des = np.array(top_5[step])[:, 1]

        for step, label in enumerate(top_5_label):
            specific_label = val_perplexity[(val_perplexity["label"] == label)]

            if x_or_c == "x":
                specific_label = specific_label.sort_values(
                    by=['x_perplexity'])
            elif x_or_c == "c":
                specific_label = specific_label.sort_values(
                    by=['c_perplexity'])
            lowest_img_filename = validation_folder + \
                specific_label.iloc[0]["filename"]
            if not os.path.exists(folder_path + top_5_label_des[step] + ".jpeg"):
                shutil.copy(lowest_img_filename, folder_path)
                os.rename(folder_path + specific_label.iloc[0]["filename"],
                          folder_path + top_5_label_des[step] + ".jpeg")


def get_same_label_similar_images(all_img_name, img_gt, x_perplexity, c_perplexity, how_many_pics=5):

    current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    validation_folder = current_path + "ILSVRC2012_img_val/"
    val_perplexity = pd.read_csv("perplexity.csv")

    for img_idx, img_name in enumerate(all_img_name):
        img = img_name.split(".")[0]
        all_records = val_perplexity[(
            val_perplexity["label"] == img_gt[img_idx])]

        records_x_perplexity = all_records["x_perplexity"].to_numpy()
        records_c_perplexity = all_records["c_perplexity"].to_numpy()

        x_perplexity_diff = np.abs(
            records_x_perplexity - x_perplexity[img_idx])
        c_perplexity_diff = np.abs(
            records_c_perplexity - c_perplexity[img_idx])

        all_records["x_perplexity_diff"] = x_perplexity_diff
        all_records["c_perplexity_diff"] = c_perplexity_diff

        all_records = all_records.sort_values(by=['x_perplexity_diff'])
        closed_x_filenames = all_records["filename"][:how_many_pics].to_numpy()
        closed_x_values = all_records["x_perplexity_diff"][:how_many_pics].to_numpy(
        )

        all_records = all_records.sort_values(by=['c_perplexity_diff'])
        closed_c_filenames = all_records["filename"][:how_many_pics].to_numpy()
        closed_c_values = all_records["c_perplexity_diff"][:how_many_pics].to_numpy(
        )

        collection = [closed_x_filenames, closed_c_filenames]

        for n, j in enumerate(collection):
            for step, i in enumerate(j):
                if n == 0:
                    folder_name = "similar_x_perplexity"
                    per_value = closed_x_values[step]
                elif n == 1:
                    folder_name = "similar_c_perplexity"
                    per_value = closed_c_values[step]
                folder_path = current_path + "BatchTestSampleImages/" + img + "/" + \
                    "confusion_class_samples/%s/" % folder_name
                os.makedirs(folder_path, exist_ok=True)
                file_name = folder_path + \
                    "no_%d_diff_%.3f.jpeg" % (step, per_value)
                if not os.path.exists(file_name):
                    shutil.copy(validation_folder + i, folder_path)
                    os.rename(folder_path + i, file_name)


def get_current_location():
    return os.path.dirname(os.path.realpath(__file__)) + "/"


def display_images(title, folder_path):
    image_paths = []
    image_names = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            image_paths.append(os.path.join(path, name))
            image_names.append(name)

    if title in ["Similar C perplexity images", "Similar X perplexity images"]:
        images_diff_label = list(
            map(lambda x: re.sub('[^0-9]', '', x[0:6]), image_names))
        images_diff_label = np.array(images_diff_label, dtype=int)
        sorted_idx = np.argsort(images_diff_label)
        image_paths = np.array(image_paths)[sorted_idx]
        image_names = np.array(image_names)[sorted_idx]

    how_many_images = len(image_names)

    row = np.floor(np.sqrt(how_many_images))
    col = np.ceil(how_many_images/row)

    if col > 3:
        col = 3
        row = np.ceil(how_many_images/col)

    fig = plt.figure(figsize=(col*4, row*4))
    fig.suptitle(title)
    for k in range(how_many_images):
        image_data = plt.imread(image_paths[k])
        ax = fig.add_subplot(row, col, k+1)
        ax.set_axis_off()
        ax.set_title(image_names[k])
        ax.imshow(image_data)

    plt.show()


# if __name__ == "__main__":
#     current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
#     instruction = "This software is used to estimate the X and C perplexities for a given image.\n \
#         Please save your images on root folder. Then input the image names seperating by space\n \
#         eg: sample1.jpg sample2.jpg sample3.jpg\n \
#         Then, input their label names according to ImageNet labels respectively\n \
#         eg: n09229709, n09229709, n09428293\n \
#         If you only want to test C perplexity, please input: no label\n \
#         Then choose the faster testing mode or accurate testing mode\n \
#         For faster testing mode, please input: 1\n \
#         For accurate testing mode, please input: 2"
#     print(instruction)
#     net_name_list = [
#         ["ResNet50"],
#         ["ResNet101"],
#         ["DenseNet121"],
#         ["DenseNet169"],
#         ["DenseNet201"],
#         ["EfficientNetB2"],
#         ["EfficientNetB0"],
#         ["Xception"],
#         ["InceptionV3"],
#         ["VGG16"],
#     ]

#     net_name_list = net_name_list[::-1]
#     # img_name = ["sample1.jpg", "sample2.jpg", "sample3.jpg"]
#     # img_gt = ["n09229709", "n09229709", "n09428293"]
#     img_name = input("Please input the image names, seperate by space: ")
#     img_gt = input("Please input the image labels, seperate by space: ")
#     test_mode = input(
#         "Please input the image labels. 1 is faster, 2 is accurate: ")
#     test_mode = int(test_mode)
#     img_name = img_name.split(" ")

#     only_c = 0
#     if img_gt == "no label":
#         image_number = len(img_name)
#         img_gt = ["n09229709"] * image_number
#         only_c = 1
#     else:
#         img_gt = img_gt.split(" ")

#     if os.path.exists(current_path + "BatchTestRecord"):
#         shutil.rmtree(current_path + "BatchTestRecord")
#     if os.path.exists(current_path + "BatchTestSampleImages"):
#         shutil.rmtree(current_path + "BatchTestSampleImages")
#     if only_c == 1:
#         if os.path.exists(current_path + "BatchTestResult"):
#             shutil.rmtree(current_path + "BatchTestResult")

#     multi_thread_optimize_test(img_name, net_name_list, test_mode)
#     top_5_c, top_5_x, x_perplexity, c_perplexity = get_perplexity(
#         img_name, img_gt, only_c, 10)
#     get_example_images(img_name, top_5_c, "c")
#     if only_c != 1:
#         get_example_images(img_name, top_5_x, "x")
#         get_same_label_similar_images(img_name, img_gt, x_perplexity, c_perplexity)


# sample1.jpg sample2.jpg sample3.jpg sample4.jpg sample5.jpg
# n09229709 n09229709 n09428293 n01883070 n02112706
