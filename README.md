# Estimation of Example Perplexity

Some examples are easier for humans to classify than others. The same should be true for deep neural networks (DNNs). We use the term **Example Perplexity** to refer to the level of difficulty of classifying an example. In this work, we propose a method to measure the perplexity of an example and investigate what factors contribute to high example perplexity.

To estimate the perplexity of an image for DNNs, we create a population of DNN classifiers with varying architectures and trained on data of varying sample sizes, just as different people have different IQs and different amounts of experiences. For an unlabeled example, the average entropy of the output probability distributions of the classifiers is taken to be the **C-perplexity** of the example, where C stands for "confusion". For a labeled example, the fraction of classifiers that misclassify the example is taken to be the **X-perplexity** of the example, where X stands for "mistake".


-----------------------------------------------------------------------------------------------------------------------
### A. Definition to the X-Perplexity and C-Perplexity

Let ![equation](https://latex.codecogs.com/svg.image?\mathit{C})  be a population of ![equation](https://latex.codecogs.com/svg.image?N) classifiers for classifying examples into ![equation](https://latex.codecogs.com/svg.image?M) classes. For a given example ![equation](https://latex.codecogs.com/svg.image?\mathbf{x}), ![equation](https://latex.codecogs.com/svg.image?\small&space;P_i(y|\mathbf{x})) is the probability distribution over the ![equation](https://latex.codecogs.com/svg.image?M) classes computed by classifier ![equation](https://latex.codecogs.com/svg.image?i).  

We define the **C-perplexity**  of an unlabelled example ![equation](https://latex.codecogs.com/svg.image?\mathbf{x}) w.r.t ![equation](https://latex.codecogs.com/svg.image?\mathit{C}) to be the following geometric mean:

![equation](https://latex.codecogs.com/svg.image?\large&space;\Phi_{C}(\mathbf{x})&space;=&space;&space;[\prod_{i=1}^N&space;2^{&space;H(P_i(y|\mathbf{x}))}]^{\frac{1}{N}})
 
The minimum possible value of C-perplexity is 1. High C-perplexity value indicates that the classifiers have low confidence when classifying the example.
 
We define the **X-perplexity** of an labelled example ![equation](https://latex.codecogs.com/svg.image?(\mathbf{x},&space;y)) w.r.t ![equation](https://latex.codecogs.com/svg.image?\mathit{C}) to be:

![equation](https://latex.codecogs.com/svg.image?\large&space;\Phi_{X}(\mathbf{x})&space;=&space;\frac{1}{N}&space;\sum_{i=1}^N&space;\mathbf{1}(C_i(\mathbf{x})&space;\neq&space;y)),

where
![equation](https://latex.codecogs.com/svg.image?\small&space;C_i(\mathbf{x})&space;=&space;\arg&space;\max_{y}&space;P_i(y|\mathbf{x})) is the class assignment function,  and ![equation](https://latex.codecogs.com/svg.image?\mathbf{1}) is the indicator function.  In words, it is the fraction of the classifiers that misclassifies the example, hence is between 0 and 1.

For the details, please check out our paper [URL].

-----------------------------------------------------------------------------------------------------------------------

### B. Our Created Classifier Population

Our created classifier population includes 500 models and can be downloaded from:

https://drive.google.com/file/d/11e49_vwygXkIUG7v9vDxy30kY7xMMby2/view?usp=sharing

The models are stored in .h5 format.

-----------------------------------------------------------------------------------------------------------------------

### C. The Source Codes for computing *X-Perplexity* and *C-Perplexity* [(Validation_Images_Test)](https://github.com/vaynexie/Example-Perplexity/tree/main/Validation_Images_Test)

Code for computing the X-Perplexity, C-Perplexity, top 5 X-confusion classes and top 5 C-confusion classes for images in ImageNet validation set. For the details about how to use the code, please check the ReadMe in the code folder.




-----------------------------------------------------------------------------------------------------------------------
### D. Result: Perplexity values for the images in the ImageNet validation set

<sub><sup>**The Perplexity.csv can be found here, for other resulted files, since the they are too large, can be downloaded by the url:https://drive.google.com/file/d/1IFi-qytTVEFSpTy-jEWaMKtc8lCuGYPv/view?usp=sharing**</sub></sup>

##### D1. Perplexity Viewer 
呈现计算所得ImageNet Validation Set中的样本分类难度和标签区分难度的整体分布情况，同时交互式展示单个样本或标签的信息。

Link: http://xai.cse.ust.hk:5000/site/index.html

<img src="https://user-images.githubusercontent.com/69588181/113534528-72898d80-957d-11eb-8d02-fd0855891d25.png" height="430" width="600">


##### D2. Perplexity.csv

The .csv file includes the following columns:

<sub><sup>file_name: file_name for the image</sub></sup>

<sub><sup>label: hand-annotated label for the image</sub></sup>

<sub><sup>c_perplexity: output C-Perplexity for this image</sub></sup>

<sub><sup>x_perplexity: output X-Perplexity for this image</sub></sup>

<sub><sup>x_perplexity_label: Discretization labels on the x_perplexity (labels include '0','0-0.1','0.1-0.2',...,'1')</sub></sup>

<sub><sup>top_5_c_perplexity_class: top 5 confusion class for C-Perplexity (includes label_id and C-Confusion Index )</sub></sup>

<sub><sup>top_5_x_perplexity_class: top 5 confusion class for X-Perplexity (includes label_id and X-Confusion Index )</sub></sup>


##### D3. netwise_c_perplexity.csv

The .csv file record the entropy for each image on ImageNet Validation Set in each model. Totally, there are 501 columns in this .csv file

<sub><sup>the first column is for the filename of the image</sub></sup>
<sub><sup>the resting 500 columns is for the 500 models</sub></sup>


##### D4. netwise_x_perplexity.csv

The .csv file record the confusion for each image on ImageNet Validation Set in each model. Totally, there are 501 columns in this .csv file.

<sub><sup>the first column is for the filename of the image</sub></sup>
<sub><sup>the resting 500 columns is for the 500 models</sub></sup>
