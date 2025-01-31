# Estimation of [Example Perplexity](https://arxiv.org/abs/2203.08813)

Some examples are easier for humans to classify than others. The same should be true for deep neural networks (DNNs). We use the term **Example Perplexity** to refer to the level of difficulty of classifying an example. In this work, we propose a method to measure the Perplexity of an example and investigate what factors contribute to high example perplexity.

To estimate the perplexity of an image for DNNs, we create a population of DNN classifiers with varying architectures and trained on data of varying sample sizes, just as different people have different IQs and different amounts of experiences. For an unlabeled example, the average entropy of the output probability distributions of the classifiers is taken to be the **C-Perplexity** of the Example, where C stands for "confusion". For a labeled example, the fraction of classifiers that misclassify the Example is taken to be the **X-Perplexity** of the Example, where X stands for "mistake".


-----------------------------------------------------------------------------------------------------------------------
## A. Definition to the X-Perplexity and C-Perplexity

Let ![equation](https://latex.codecogs.com/svg.image?\mathit{C})  be a population of ![equation](https://latex.codecogs.com/svg.image?N) classifiers for classifying examples into ![equation](https://latex.codecogs.com/svg.image?M) classes. For a given example ![equation](https://latex.codecogs.com/svg.image?\mathbf{x}), ![equation](https://latex.codecogs.com/svg.image?\small&space;P_i(y|\mathbf{x})) is the probability distribution over the ![equation](https://latex.codecogs.com/svg.image?M) classes computed by classifier ![equation](https://latex.codecogs.com/svg.image?i).  

We define the **C-Perplexity**  of an unlabelled example ![equation](https://latex.codecogs.com/svg.image?\mathbf{x}) w.r.t ![equation](https://latex.codecogs.com/svg.image?\mathit{C}) to be the following geometric mean:

![equation](https://latex.codecogs.com/svg.image?\large&space;\Phi_{C}(\mathbf{x})&space;=&space;&space;[\prod_{i=1}^N&space;2^{&space;H(P_i(y|\mathbf{x}))}]^{\frac{1}{N}})
 
The minimum possible value of C-perplexity is 1. High C-perplexity value indicates that the classifiers have low confidence when classifying the Example.
 
We define the **X-perplexity** of an labelled example ![equation](https://latex.codecogs.com/svg.image?(\mathbf{x},&space;y)) w.r.t ![equation](https://latex.codecogs.com/svg.image?\mathit{C}) to be:

![equation](https://latex.codecogs.com/svg.image?\large&space;\Phi_{X}(\mathbf{x})&space;=&space;\frac{1}{N}&space;\sum_{i=1}^N&space;\mathbf{1}(C_i(\mathbf{x})&space;\neq&space;y))

, where
![equation](https://latex.codecogs.com/svg.image?\small&space;C_i(\mathbf{x})&space;=&space;\arg&space;\max_{y}&space;P_i(y|\mathbf{x})) is the class assignment function,  and ![equation](https://latex.codecogs.com/svg.image?\mathbf{1}) is the indicator function.  In words, it is the fraction of the classifiers that misclassifies the example, hence is between 0 and 1.

For the details, please check out our [paper](https://arxiv.org/abs/2203.08813).

-----------------------------------------------------------------------------------------------------------------------

## B. Our Created Classifier Population

Our created classifier population includes 500 models and can be downloaded from:

https://drive.google.com/file/d/11e49_vwygXkIUG7v9vDxy30kY7xMMby2/view?usp=sharing .

The models are stored in .h5 format and can be read in by *keras.models.load_model()*.

-----------------------------------------------------------------------------------------------------------------------

## C. The Source Codes for computing *X-Perplexity* and *C-Perplexity* [(Validation_Images_Test)](https://github.com/vaynexie/Example-Perplexity/tree/main/Validation_Images_Test)

Codes for computing the X-Perplexity, C-Perplexity, top 5 X-confusion classes and top 5 C-confusion classes for images in ImageNet validation set, are under the folder [(Validation_Images_Test)](https://github.com/vaynexie/Example-Perplexity/tree/main/Validation_Images_Test). For the details about using the code, please check the ReadMe in the code folder.




-----------------------------------------------------------------------------------------------------------------------
## D. Result: Perplexity values for the images in the ImageNet validation set

### [D1. Perplexity Viewer](http://xai.cse.ust.hk:5000/site/index.html) 
An interactive visual viewer to show the overall distribution of X-Perplexity and C-Perplexity of the ImageNet Validation Set, along with information about individual samples.

<img src="https://user-images.githubusercontent.com/69588181/113534528-72898d80-957d-11eb-8d02-fd0855891d25.png" height="430" width="600">


### [D2. Perplexity.csv](https://github.com/vaynexie/Example-Perplexity/blob/main/perplexity.csv)

The .csv file includes the the compuated perplexity information for the images in ImageNet validation set:

<sub>file_name: file_name for the image</sub>

<sub>label: hand-annotated label for the image</sub>

<sub>c_perplexity: output C-Perplexity for this image</sub>

<sub>x_perplexity: output X-Perplexity for this image</sub>

<sub>x_perplexity_label: Discretization labels on the x_perplexity (labels include '0','0-0.1','0.1-0.2',...,'1')</sub>

<sub>top_5_c_perplexity_class: top 5 confusion class for C-Perplexity (includes label_id and C-Confusion Index )</sub>

<sub>top_5_x_perplexity_class: top 5 confusion class for X-Perplexity (includes label_id and X-Confusion Index )</sub>


### [D3. netwise_c_perplexity.csv](https://drive.google.com/file/d/1IFi-qytTVEFSpTy-jEWaMKtc8lCuGYPv/view?usp=sharing)

The .csv file record the entropy for each image on ImageNet Validation Set in each model. Totally, there are 501 columns in this .csv file

The first column is for the images' filenames, while the rest 500 columns are for the 500 models indicating the confusion of the images under the models.


### [D4. netwise_x_perplexity.csv](https://drive.google.com/file/d/1IFi-qytTVEFSpTy-jEWaMKtc8lCuGYPv/view?usp=sharing)

The .csv file record the confusion for each image on ImageNet Validation Set in each model. Totally, there are 501 columns in this .csv file.

The first column is for the images' filenames, while the rest 500 columns are for the 500 models indicating the confusion of the images under the models.


-----------------------------------------------------------------------------------------------------------------------
Enquiry for the code related issues:
Vayne Xie (wxieai@connect.ust.hk) (The Hong Kong University of Science and Techonology)
