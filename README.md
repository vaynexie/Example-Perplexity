# HKUST-Huawei Collaborative Research Project: YB2015070037
## WP1 Estimation of Example Perplexity

### 技术框架

<img src="https://user-images.githubusercontent.com/69588181/113399272-6a96d700-934c-11eb-879e-a33f979ed7ef.png" height="400" width="600">


-----------------------------------------------------------------------------------------------------------------------
#### A. 模型池的训练

<img src="https://user-images.githubusercontent.com/69588181/113399432-b2b5f980-934c-11eb-8b81-b7e579b46bf3.png" height="430" width="600">

The training data and classifier population can be downloaded from:

ImageNet Validation Set （训练数据）: 

https://drive.google.com/file/d/1p4TpYtK5MX2I_H_s8jzOHde7IBsDM-2v/view?usp=sharing

Classifier Population (模型池——训练得到的500个模型）: 

https://drive.google.com/file/d/11e49_vwygXkIUG7v9vDxy30kY7xMMby2/view?usp=sharing

-----------------------------------------------------------------------------------------------------------------------
#### B. The Source Code for computing *Example Perplexity (X-Perplexity and C-Perplexity)* 

<img src="https://user-images.githubusercontent.com/69588181/113534016-ed51a900-957b-11eb-848a-0f2d163a4318.png" height="430" width="600">



For the details about how to use the code, please check the ReadMe in the code folder

##### B1. Validation_Images_Test

<sub><sup>**Code for computing the X-Perplexity, C-Perplexity, top 5 X-confusion classes and top 5 C-confusion classes for image in ImageNet validation set**</sub></sup>


##### B2. Multiple_Images_Test

<sub><sup>**Code for computing the X-Perplexity, C-Perplexity, top 5 X-confusion classes and top 5 C-confusion classes, and providing example explanation images for the input images (the input images do not need to be in the ImageNet validation set, can be any image from the Internet as long as its label is in the 1000 classes of ImageNet.** </sub></sup>

##### B3. Input_different

<sub><sup>**Code for allowing user to compute image perplexity using their own classifier populaiton on different datasets apart from ImageNet.** </sub></sup>

-----------------------------------------------------------------------------------------------------------------------
#### C. Result: Perplexity values for the images in the ImageNet validation set

<sub><sup>**The Perplexity.csv can be found here, for other resulted files, since the they are too large, can be downloaded by the url:https://drive.google.com/file/d/1IFi-qytTVEFSpTy-jEWaMKtc8lCuGYPv/view?usp=sharing**</sub></sup>

##### C1. Perplexity Viewer 
呈现计算所得ImageNet Validation Set中的样本分类难度和标签区分难度的整体分布情况，同时交互式展示单个样本或标签的信息。

Link: http://xai.cse.ust.hk:5000/site/index.html

<img src="https://user-images.githubusercontent.com/69588181/113534528-72898d80-957d-11eb-8d02-fd0855891d25.png" height="430" width="600">


##### C2. Perplexity.csv

<sub><sup>**The .csv file includes the following columns:**</sub></sup>

<sub><sup>**file_name: file_name for the image**</sub></sup>

<sub><sup>**label: hand-annotated label for the image**</sub></sup>

<sub><sup>**c_perplexity: output C-Perplexity for this image**</sub></sup>

<sub><sup>**x_perplexity: output X-Perplexity for this image**</sub></sup>

<sub><sup>**x_perplexity_label: Discretization labels on the x_perplexity (labels include '0','0-0.1','0.1-0.2',...,'1')**</sub></sup>

<sub><sup>**top_5_c_perplexity_class: top 5 confusion class for C-Perplexity (includes label_id and C-Confusion Index )**</sub></sup>

<sub><sup>**top_5_x_perplexity_class: top 5 confusion class for X-Perplexity (includes label_id and X-Confusion Index )**</sub></sup>
**

##### C3. netwise_c_perplexity.csv

<sub><sup>**The .csv file record the entropy for each image on ImageNet Validation Set in each model**</sub></sup>

<sub><sup>**Totally, there are 501 columns in this .csv file**</sub></sup>

<sub><sup>**the first column is for the filename of the image**</sub></sup>
<sub><sup>**the resting 500 columns is for the 500 models**</sub></sup>


##### C4. netwise_x_perplexity.csv

<sub><sup>**The .csv file record the confusion for each image on ImageNet Validation Set in each model**</sub></sup>

<sub><sup>**Totally, there are 501 columns in this .csv file**</sub></sup>

<sub><sup>**the first column is for the filename of the image**</sub></sup>
<sub><sup>**the resting 500 columns is for the 500 models**</sub></sup>