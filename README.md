# Density-Estimation-and-classification
In this project, I performed  parameter estimation for a given dataset (which is a subset from the MNIST dataset)

## Dataset
The total number of samples in the training set for digit 7 is 6265 and for digit 8 is
5851. The total number of samples for the testing set for digits 7 and 8 is 1028 and
974 respectively. Each MNIST image of a digitised picture of a single handwritten
digit character .Each image is 28x28 in size.So,there are a total of 784 pixels per
image.ie.one sample has 784 features.Just  go to the original MNIST dataset (available here http://yann.lecun.com/exdb/mnist/ ) to extract the images for digit 7 and digit 8, to form the dataset for this project

## Methodology 
### Feature Extraction
For the ease of classification,we are extracting two features from the dataset, i.e,,I have 
 reduced the feature dimension to 2.First one is the average of all pixel values in
the image(Feature 1) and second one is the standard deviation of all pixel values in
the image(Feature 2).
### NAIVE BAYES CLASSIFICATION
We assume that the 2 features extracted are independent and that each image is
drawn from a 2-d normal distribution.

We need to calculate the prior probability of class and posterior probability of the
sample in a given class from the training dataset.For calculating the posterior
probability we need to estimate 8 different parameters,i.e. The mean and standard
deviation of each feature with respect to each class.The values are shown in the
table below .

| Parameters  | Digit 7 | Digit 8     |
|-------|-----|----------|
| MEAN FOR FEATURE 1 |0.1145| 0.1501|
| STD FOR FEATURE 1 |0.0306| 0.0386|
|MEAN FOR FEATURE 2| 0.2877  | 0.3206|
|STD FOR FEATURE 2| 0.0382 |0.0399  |


We will be able to get a bi-variate gaussian distribution for each digit .Since the features are independent we can multiply the normal distribution of each feature to get the bi-variate distribution.The expression for calculating the bi-variate normal distribution is given below(y is class label ,x is the sample and $ x_{1},x_{2} $ are the two features for each sample x)

$$ p( x_{1},x_{2}|y=given class) = \frac{e^{-\frac{1}{2}[\frac{(x_{1}-\mu_{1})^2}{\sigma_{1}^2}+\frac{(x_{2}-\mu_{2})^2}{\sigma_{2}^2}]}}{2\pi\sigma_{1}\sigma_{2}} $$

Here for a given class we are finding the joint probability density of the two features.
$\mu_{1},\mu_{2}$ are the mean of the two features of the given class and $\sigma_{1},\sigma_{2}$ are the standard deviation of the two features

We now use the testing data,which is also to reduced to 2d feature space, to calculate these joint probability w.r.t two classes and multiply it with the prior probability of the class i.e.𝑝(𝑦).By bayes theorem the posterior probability of class
where the sample is given 𝑝(𝑦|𝑥) is proportional to the product of 𝑝(𝑥|𝑦) and 𝑝(𝑦).So
we will get two probability of a given sample.The sample belong to the class which
has the highest posterior probability 𝑝(𝑦|𝑥)

### LOGISTIC REGRESSION
This is a linear classifier.We are using the original training dataset, which has 784 features,for training logistic regression(LG) model.We define a weight value for each feature and use the gradient ascent algorithm to update the weights by trying to
maximise the likelihood.We are updating the weights independently.The expression
for gradient ascent is given below
$$ w^{(k+1)} = w^{(k)}+\eta\nabla_{w^{(k)}}l(w) $$
The $l(w)$ is the log likelihood function and $\eta$ is the learning rate.We used 0.0001 as
the learning rate .Total number of iteration,while training the model , for updating the
weights is 10000
We need to append the value 1 in the first columns of all the rows in the training and
test dataset .We will get the value of the weights by using the training dataset.Then
we multiply the test data with the weights and check whether the value of the product
of each sample point with weight is greater than 0.If it is true then the label of that
particular sample point is 1(digit 8) else 0(digit 7)
## Results
For computing the prediction accuracy of the models I used the sklearn module and
computed the accuracy and the confusion matrix for each of the models.The total
prediction accuracy and accuracy for predicting each digit of each model is in the
table given below.

| CLASSIFICATION ACCURACY |NAIVE BAYES (%) |LOGISTIC REGRESSION(%)|
|-------|-----|----------|
|DIGIT 7| 69.08| 98.93|
|DIGIT 8| 68.65| 99.27|
|TOTAL |68.88| 99.10|

As we can see, the prediction accuracy of logistic regression model is greater than
Naive Bayes model.The logistic regression model perform well because of less strict
assumptions made on the data  compared to the NB model