# KNN From Scratch

In this project I created the K-Nearest Neighbors (KNN) Algorithm from scratch using Python and the MNIST handwritten digit data set from Tensor Flow’s Dataset module. This dataset contains 70,000 handwritten digits, 60,000 for training and 10,000 for testing, along with their respective labels. 
KNN is a classification method that takes a test sample and calculates the distance between all the training samples. The k nearest training samples are taken into consideration and the majority label is assigned to the test sample. 

This simple algorithm is surprisingly accurate, especially with the handwritten digit data set. In fact, the error rate is at max two times the Bayes error[1]. 

![alt text](https://github.com/CodeAspect/KNNFromScratch/blob/main/KNNError.png?raw=true)

In this implementation of the algorithm, I used Sci-Kit Learns pairwise_distances function to calculate the Euclidean distance between images for all test cases, stored the top 60, then tested different k values. The values chosen for k were 1, 3, 5, 10, 20, 30, 40, 50, 60 which resulted in a max classification accuracy of about 97.05 with the k = 3 value. Below is a chart of Accuracy results per k value. 

![alt text](https://github.com/CodeAspect/KNNFromScratch/blob/main/KNNResults.png?raw=true)

The drawback to this algorithm is that it is extremely costly to implement. To find the nearest training values, every single training value’s distance must be measured. If you have an extremely large data set this can be prohibitive. Some solutions I used to work around this problem were using the pairwise_distance function which makes use of parallelism when computing distances. Also, instead of recomputing the entire test set for every k, I computed it entirely and reused the distances for various k values. 
My conclusions are that KNN is a powerful and simple algorithm best used for classifying potentially similar items such as handwritten digits. 

References:

[1] Duda, Richard O. and Hart, Peter E.  Pattern classification and scene analysis / Richard O. Duda, Peter E. Hart  Wiley New York  1973 
