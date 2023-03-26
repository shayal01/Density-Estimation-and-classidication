# Density-Estimation-and-classification
In this project, I performed  parameter estimation for a given dataset (which is a subset from the MNIST dataset)

## Dataset
The total number of samples in the training set for digit 7 is 6265 and for digit 8 is
5851. The total number of samples for the testing set for digits 7 and 8 is 1028 and
974 respectively. Each MNIST image of a digitised picture of a single handwritten
digit character .Each image is 28x28 in size.So,there are a total of 784 pixels per
image.ie.one sample has 784 features.

## Methodology && Results
### Feature Extraction
For the ease of classification,we are extracting two features from the dataset, i.e,,I have 
 reduced the feature dimension to 2.First one is the average of all pixel values in
the image(Feature 1) and second one is the standard deviation of all pixel values in
the image(Feature 2).

I build a Naive Bayes and Logistic Regression classifiers without using packages like sklearn for computing the boundaries.
 Results are computed and can be found here
