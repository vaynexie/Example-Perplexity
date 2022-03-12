# Readme: Multiple_Images_Test

The propose of this code file is to allow user to upload several new images and their labels, in order to obtain the X-Perplexity, C-Perpelxity, 
C-confusion classes, X-confusion classes for these new images (the image can be any image from the Internet as long as its class label is in the ImageNet). 


Note that: There are totally 1000 classes in ImageNet, for detailed classes, can find in the file 'imagenet_class_index.json'. 
When input the labels into our code, we require the label is in the form of **ImageNet class index
(for example, need to input "n01440764" for class "tench")**

For each input image, the code will output:
1. The X-Perplextiy, C-Perplexity, Top 5 X-Confusion classes and Top 5 C-Confusion classes for the input images;
2. Five images in the same label class with similar X-Perplexity and C-Perplexity value in ImageNet validation set;
3. Example images for the top confusion classes.

The images output in 2 and 3 can be used to explain and understand why the input images are easily to be confused with other classes.  

## A. Files that need to download and decompress into this folder for executing the codes:

ImageNet Validation Set: https://drive.google.com/file/d/1p4TpYtK5MX2I_H_s8jzOHde7IBsDM-2v/view?usp=sharing

Classifier Population: https://drive.google.com/file/d/11e49_vwygXkIUG7v9vDxy30kY7xMMby2/view?usp=sharing




## B. An example for using this code

**Before running this code on the new images,please put the images in this code folder.**

* If you want to run the testing code in Python directly, please run **test_multiple_images.py**, the resulted example images will be saved in the code folder;
* You can aslo run the **test_multiple_images.ipynb** in Jupyter Notebook, then the resulted perplexity and example images will be shown in the notebook directly.

when the code is runned, the system will require user to input the image name and label, and choose the testing mode:

```
This software is used to estimate the X and C perplexities for a given image.  
         Please save your images on root folder. Then input the image names seperating by space. 
         eg: sample1.jpg sample2.jpg sample3.jpg. 
         Then, input their label names according to ImageNet labels respectively. 
         eg: n09229709, n09229709, n09428293. 
         If you only want to test C perplexity, please input: no label. 
         Then choose the faster testing mode or accurate testing mode. 
         For faster testing mode, please input: 1. 
         For accurate testing mode, please input: 2. 
         
Please input the image names, seperate by space: screwdriver.jpg
Please input the image labels, seperate by space: n04154565
Please input the image labels. 1 is faster, 2 is accurate: 1
```

As you can see in the code, there are some available options:
* If the image is with label, you can input the image with the label into the system, and the code will output both X-Perplexity and C-Perpelxity. If the label is not available, you can also input the image only, then the code will only output the X-Perplexity;
* There are two modes for running the code:

mode 1: Only use the selected 100 models to produce X/C perplexity, which will be faster (as we checked, the perplexity result output by these 100 models will not be much different from that output the whole 500 models in the classifier population in most of cases);  

mode 2: Use all the 500 models in the classifier population to produce X/C perplexity.

Here we input a image with label n04154565 (screwdriver) into the system, the output result is shown below:


<img src="/Multiple_Images_Test/BatchTestSampleImages/screwdriver.jpg" height="300" width="300">


```
true label is ['n04154565'] ['screwdriver']
x perplexity is [0.02325581]
c perplexity is [1.15888804]
top 5 c confusion classes are [[('n03970156', 'plunger', 0.068), ('n04154565', 'screwdriver', 0.053), ('n02790996', 'barbell', 0.016), ('n03476684', 'hair_slide', 0.011), ('n04367480', 'swab', 0.009)]]
top 5 x confusion classes are [[('n04154565', 'screwdriver', 0.977), ('n03970156', 'plunger', 0.012), ('n03729826', 'matchstick', 0.012)]]
```

Also, the output example images for the input image is placed in the directory /BatchTestSampleImages/screwdriver/confusion_class_samples:
* the sub-folder 'x': includes example images for the top 5 X-confusion classes;  
* the sub-folder 'c': includes example images for the top 5 C-confusion classes;  
* the sub-folder 'similar_x_perplexity': includes five example images from class 'screwdriver' with similar X-perplexity in the ImageNet Validation set;
* the sub-folder 'similar_c_perplexity': includes five example images from class 'screwdriver' with similar C-perplexity in the ImageNet Validation set (file name no_0_diff_0.027.png-->'no_0' means the closest one and 'diff_0.027' means the absolute difference value between the input image and the example image is 0.027).


