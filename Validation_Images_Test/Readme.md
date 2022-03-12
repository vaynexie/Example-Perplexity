# Readme: Validation_Images_Test

Run the classifier population on the ImageNet Validation set to produce:

* The prediction logits for the images under each model (500 .npy files)
* The X-Perplextiy, C-Perplexity, Top 5 X-Confusion classes and Top 5 C-Confusion classes for each image

## A. Files that need to download and decompress into this folder for executing the codes:

ImageNet Validation Set: https://drive.google.com/file/d/1p4TpYtK5MX2I_H_s8jzOHde7IBsDM-2v/view?usp=sharing

Classifier Population: https://drive.google.com/file/d/11e49_vwygXkIUG7v9vDxy30kY7xMMby2/view?usp=sharing


-------------------------------------------------------------------------------------------------------------


## B. Code 1: test.py

The purpose of this code is to run the 500 trained models on ImageNet Validation Set and output the prediction result.


### Input: 

a. 500 trained model (in the form of .h5) 

b. Folder ILSVRC2012_img_val: Totally 50000 images in the form of .JPEG from ImageNet Validation Set

c. ILSVRC2012_validation_ground_truth.txt: Hand-annotated label (in the form of label index) for the images in ImageNet Validation Set 

d. meta.mat: Dictionary that includes detailed information for labels (includes label_index, label_id, label_name, label_explanation)


### Output:
Totally 500 .npy files that record the prediction results in the form of logits for each images under each model 


**Note that: Need to run Code 1 first and then run Code2**

(But the running of Code 1 is time consuming, you may consider using the 500 .npy files we have produced, can be download in https://drive.google.com/file/d/1K2_k2KrKNDVK035uts9EHXw_BoEPAYFz/view?usp=sharing)


-------------------------------------------------------------------------------------------------------------

## C. Code 2: c_x_perplexity_and_top_5.py

### Input:

a. 500 .npy files that record the prediction results  

b. Folder ILSVRC2012_img_val: Totally 50000 images in the form of .JPEG from ImageNet Validation Set

c. ILSVRC2012_validation_ground_truth.txt: Hand-annotated label (in the form of label index) for the images in ImageNet Validation Set 

d. meta.mat: Dictionary that includes detailed information for labels (includes label_index, label_id, label_name, label_explanation)


### Output:

**a. Perplexity.csv**

The .csv file includes the following columns:

file_name: file_name for the image;  

label: hand-annotated label for the image;   

c_perplexity: output C-Perplexity for this image;  

x_perplexity: output X-Perplexity for this image;     

x_perplexity_label: Discretization labels on the x_perplexity (labels include '0','0-0.1','0.1-0.2',...,'1');  

top_5_c_perplexity_class: top 5 confusion class for C-Perplexity (includes label_id and C-Confusion Index );  
 
top_5_x_perplexity_class: top 5 confusion class for X-Perplexity (includes label_id and X-Confusion Index ). 

<br/>

<br/>


**b. netwise_c_perplexity.csv**

The .csv file record the entropy for each image on ImageNet Validation Set in each model

Totally, there are 501 columns in this .csv file

the first column is for the filename of the image
the resting 500 columns is for the 500 models

<br/>

<br/>

**c. netwise_x_perplexity.csv**

The .csv file record the confusion for each image on ImageNet Validation Set in each model

Totally, there are 501 columns in this .csv file

the first column is for the filename of the image
the resting 500 columns is for the 500 models

(if prediction for model M1 on image I1 is corrected, then [M1,I1]=0, otherwise [M1,I1]=1) 

<br/>

<br/>

**d. x_perplexity_confusion.npy and  c_perplexity_confusion.csv**

x_perplexity_confusion is to record the top-1 classification results for the images, it is a dictionary in the form of {'nxxxxxxx':50,'nxxxxxxx':50,....},

where 'nxxxxxxx' is the label index and 50 means there are 50 times the image is classified as 'nxxxxxxx' among the 500 times of prediction (500 models); 
    
c_perplexity_confusion is to record the entropy value for each class for the images ( one model will produce -p*log(p) for each class of the image, finally we take the average on the 500 models for the image).
    

x_perplexity_confusion, c_perplexity_confusion  will also be output, but here we mainly used them for producing the Top X confusion labels and Top C confusion labels for the Image (those will be finally recorded in **a. Perplexity.csv**)

We output them in case we need them in further analysis (e.g.  x_perplexity_confusion can be used to analyze the frequent confusion pairs)
