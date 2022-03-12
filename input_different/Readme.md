# Readme: Input_different

In order to allow more feasible testing on our image perplexity estimation method, we design this code to allow user to use their classifier populaiton on different datasets apart from ImageNet.

**Note that: as mentioned in the WP1 report, the final value of perplexity is highly related to the choice of classifier population. Also, we show that the use of a population of classifiers with various strengths is necessary for obtaining robust perplextiy values. It is suggested that user can follow our logic of building the classifier population (see WP1 Report) for their dataset in order to include classifiers with various strengths.**

## Input:

1. folder predict_result that contains .npy files that record the array of predicted logits for each image under each classifier in classifier population.

If there are 10 classifiers in the population, then there should be 10 .npy files.

The array shoule be in the form of:

```
array([[7.2252315e-06, 2.9635852e-05, 2.3332255e-01, 2.1808107e-04,
        7.5596154e-01, 1.0460946e-02],
       [3.5327324e-04, 3.6028787e-03, 1.8545064e-04, 9.9569219e-01,
        5.0292609e-05, 1.1589831e-04],
       [3.7548286e-03, 3.9401129e-03, 3.4126194e-04, 9.3326890e-01,
        6.4925617e-04, 5.8045592e-02],
       ...,
       [1.6011914e-05, 1.2104671e-04, 5.5135321e-04, 1.7557220e-04,
        4.4099253e-04, 9.9869508e-01],
       [1.0288066e-03, 2.4988176e-04, 8.1854165e-01, 2.2367602e-03,
        1.6062672e-01, 1.7316177e-02],
       [9.9804509e-01, 2.3915063e-04, 6.8552158e-04, 4.0000133e-04,
        5.0650164e-04, 1.2371408e-04]], dtype=float32)
```

2. label_dict.json:
include a dictionary that records the order of classes in the output layer of the classifier

```
{'0': ['forest'],
 '1': ['buildings'],
 '2': ['glacier'],
 '3': ['street'],
 '4': ['mountain'],
 '5': ['sea']}
```

3. result_df.pkl:
include a dataframe that records the ground truth label for the dataset
 
```
 	filename	class
0	23933.jpg	forest
1	23728.jpg	forest
2	24047.jpg	forest
3	23700.jpg	forest
4	24251.jpg	forest
...	...	...
2995	21481.jpg	sea
2996	24076.jpg	sea
2997	23725.jpg	sea
2998	21898.jpg	sea
2999	23731.jpg	sea
```


## Output:

1. Perplexity.csv

The .csv file includes the following columns:

file_name: file_name for the image. 

label: hand-annotated label for the image. 

c_perplexity: output C-Perplexity for this image.

x_perplexity: output X-Perplexity for this image.  

x_perplexity_label: Discretization labels on the x_perplexity (labels include '0','0-0.1','0.1-0.2',...,'1'). 

top_5_c_perplexity_class: top 5 confusion class for C-Perplexity (includes label_id and C-Confusion Index ). 

top_5_x_perplexity_class: top 5 confusion class for X-Perplexity (includes label_id and X-Confusion Index ). 


2. netwise_c_perplexity.csv

The .csv file record the entropy for each image on ImageNet Validation Set in each model


3. netwise_x_perplexity.csv

The .csv file record the confusion for each image on ImageNet Validation Set in each model

(if prediction for model M1 on image I1 is corrected, then [M1,I1]=0, otherwise [M1,I1]=1) 

 

