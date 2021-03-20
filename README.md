# multinomial naive bayes classifier
#### Adam Dhalla [adamdhalla.com](https://adamdhalla.com/)

A multinomial naive bayes classifier similar to the ones provided by scikit-learn.

### **Input Syntax**
The file `multinomialNaiveBayes.py` contains code for a class  `multinomialNaiveBayes` which has two methods. 

The `__init__()` method requires two inputs. These are `X`, which is a (m, n) numpy ndarray of training data, and `targets`, a  (m, 1) numpy vector of associated target values. Each row of of X is a training example, and each column is the value of the _n_ features. 

By defining the classifier with training data `X` and `targets`, we are training our model. The classifier does not automatically segment our data into training and test data, that must be done by the operator. The class simply trains on the data. 

Then, we can use the only method `.fit()` which takes in a (z, n) sized numpy array of test values, where z is the amount of test examples. This can (and often is) just a single example, where our input is a row vector of size (1, n). The return value of this function is a (z, 1) sized numpy array of predictied classes for each example. 

### **Under the Hood**
_I cover in detail the way this algorithm works in an article I wrote, [here](https://adamdhalla.medium.com/naive-bayes-classifiers-ii-application-ea6faa37479b). I cover the basic theory for Naive Bayes Classifiers in another article I wrote, [here](https://medium.datadriveninvestor.com/naive-bayes-classification-i-theory-46fe87c07a6f). Check both of those out for a detailed understanding of Naive Bayes Classifiers and this specific algorithm._ 

Without going into too much details of why exactly the Naive Bayes Classifier works (which is in the articles above), I'll give an example aimed at people already familiar with Bayes Theorem etc... here. If this doesn't make sense, read the articles! 

1. During training, we essentially calculate Bayes' Rule (without the denominator term, which is a constant). For each class present in our training data, we calculate a prior (the chance of an example being in class x). 
2. Then, we calculate conditional probabilities for each feature in each class, where the class is treated as given. for each feature x in each class c, we basically calculate P(x | c).
3. We store these conditional probabilities in an array. 
4. During test time, for each example, we test each class. For each class, in each example, we multiply the values of the features in our test example by the corresponding conditional probabilities calculated during training. We append each of these products to a list.
5. For each class, we multiply all the products of the conditional probabilities and the examples' features to get a single scalar. We then multiply this scalar by the prior of the current class. This is the probability we assign to this class.
6. We then do this for each class in each example. We take the larger probability as the class assignment for that example.
7. We do this for each example and return each prediction as a list.
