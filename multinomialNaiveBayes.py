# naive bayes 
#testfile 
import pandas as pd
import numpy as np
import operator 

class MultinomialNaiveBayes():

    def __init__(self, X, targets):
        """does non-gaussian Naive Bayes classification on the inputted data.
        
        X = a numpy array of numerical values. The dimensions are (m x n), where
            each row is a training example and each column is a feature. 

        targets = a numpy array of numerical values. The associated targets with
                    each training example in X, with size (m x 1). 
        """

        # split into training examples for each class
        self.classes = np.unique(targets).tolist()
        self.amountOfClasses = len(self.classes)

        self.n = np.shape(X)[1]
        self.m = np.shape(X)[0]

        # set small number epsilon to get rid of any zeros
        self.epsilon = 0.001

        # merge X and targets for ease of indexing
        A = np.concatenate([X, targets], axis=1)

        # turn into a pandas df
        df = pd.DataFrame(A)

        #make list of split datasets based on class
        self.splitdata = {}

        for c in self.classes:
            x = df.loc[df[self.n] == c]
            self.splitdata[c] = x

        self.allLikelihoods = {}
        self.allPriors = {}
        for c in self.classes:
            # for each class, calculate the corresponding probabilities
            X = self.splitdata[c]
            M = np.shape(X)[0]
            
            # calculate the class prior probability (out of all examps, what prob is the class?)
            self.allPriors[c] = M/self.m 
    
            # calculate total amount of counts for each variable in the class
            total = X.loc[:, X.columns[:-1]].to_numpy().sum()

            # calculate all likelihood terms P(N|c) using multinomial distribution
            likelihoods = {}

            for feature in range(self.n):
                featureOccurrences = X.loc[:, X.columns[feature]].to_numpy().sum()
                
                # return +1 for stability on top and bottom
                likelihood = (featureOccurrences + 1) / (total + self.n)
                likelihoods[feature] = (likelihood)
            
            self.allLikelihoods[c] = likelihoods
        
    
    def fit(self, x):
        """fits the model to a new example.

           x: a matrix where each column is a feature, and each row is a training example. If 
              multiple training examples, answer returned will be a numpy array of predictions in 
              size ([examples, 1]). IMPORTANT: INPUT MUST NOT BE A RANK ONE ARRAY. IF SUBMITTING
              A SINGLE EXAMPLE, MUST BE IN FORMAT np.array([[1, 2, 3, 4]]).

        """
    

        # run through each calculated conditional probability and multiply it by how much it appears in 
        # new example
        
        # if single example, reshape into 2d array so we can iterate properly
        
        exampleResults = [] 

        logits = {} 
        for ex in x:
            for c in self.classes:
                runningCondProbs = []
                for feature in range(self.n):
                    currentConditionalProb = round(ex[feature]*self.allLikelihoods[c][feature], 4)
                    currentConditionalProb += self.epsilon
                    runningCondProbs.append(currentConditionalProb)
            
                unnormalizedClassProb = (np.prod(runningCondProbs))*self.allPriors[c]
                    
                logits[c] = unnormalizedClassProb
            
            assignedClass = max(logits.items(), key=operator.itemgetter(1))[0]
            exampleResults.append(assignedClass)
          
        
        # return the exampleResults as a (examples, 1) size matrix
        return ((np.array(exampleResults).reshape((np.shape(x)[0], -1))))
        
            

    
  
