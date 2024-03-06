# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

"""
This module implements a Machine Learning Classifier combining SVM and KNN classifier to make predictions for pacman.
In case of an adversarial state between the two classifiers, a final prediction is determined by a random biased selection.
"""

import numpy as np
from collections import Counter

def calculate_euclidean_distance(x1,x2):
    distance = np.sqrt(np.sum((x1-x2)**2))    
    return distance

class KNN:
    """
    K Nearest Neighbour classifier implementation. 
    """
    def __init__(self, k=5):
        self.k = k
        
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        self.X_train = X
        self.y_train = y
        
    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self,x): 
        #compute the distance
        distances = [calculate_euclidean_distance(x, x_train) for x_train in self.X_train]
        
        #get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print("KNN top 5 are -> ", k_nearest_labels)
        
        #majority vote
        most_common = Counter(k_nearest_labels).most_common()[0][0]
        print("sample: ", x, "\t move by KNN: ", [most_common])
        return most_common
    

class SVM_OvA:
    """
    Support Vector Machine One-versus-All classifier implementation. 
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.classifiers = []

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        self.classifiers = []

        for target_class in np.unique(Y):
            # Create a binary target variable for the current class
            y = np.where(Y == target_class, 1, -1)
            # Initialize and train an SVM model for the current class
            svm = BinarySVM(self.lr, self.lambda_param, self.n_iters)
            svm.fit(X, y)
            self.classifiers.append(svm)
            
    def predict(self, X):
        X = np.array(X)
        # Compute predictions for each SVM and select the class with the highest score
        predictions = np.array([svm.predict_score(X) for svm in self.classifiers]).T
        print("SVM by order are -> ", predictions)
        print("sample: ", X, "\t move by SVM: ", np.argmax(predictions, axis=1))
        return np.argmax(predictions, axis=1)


class BinarySVM:
    # This class is similar to the original SVM class provided
    # but includes a method for scoring predictions
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict_score(self, X):
        # Return the raw score rather than the sign
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        return np.sign(self.predict_score(X))


class HybridClassifier:
    """
    Hybrid Classifier fuses the decision of SVM and KNN classifiers.
    """
    def __init__(self, KNN_k=5, SVM_lr=0.001, SVM_lambda=0.01, SVM_iters=1000):
        self.knn = KNN(k=KNN_k)
        self.svm = SVM_OvA(learning_rate=SVM_lr, lambda_param=SVM_lambda, n_iters=SVM_iters)

    def fit(self, X, y):
        self.knn.fit(X, y)
        self.svm.fit(X, y)

    def predict(self, X):
        knn_predictions = self.knn.predict(X)
        svm_predictions = self.svm.predict(X)
        return self.combine_predictions(knn_predictions, svm_predictions)

    def combine_predictions(self, knn_preds, svm_preds):
        final_predictions = []
        for knn_pred, svm_pred in zip(knn_preds, svm_preds):
            if knn_pred == svm_pred:  # If both classifiers agree
                move_bycombine = knn_pred
                final_predictions.append(move_bycombine)
            else:
                move_bycombine = np.random.choice([knn_pred, svm_pred])
                final_predictions.append(move_bycombine)  
        print("Final Move: ", move_bycombine)
        return final_predictions


class Classifier:
    """
    Classifier class utilises the HybridClassifier for predicting class label.
    """
    clf = HybridClassifier()
    def __init__(self):
        print("Initialising classifier")
        self.clf = HybridClassifier()
        self.previous_state = 1 # history
        pass

    def reset(self):
        print("Resetting classifier")
        pass
    
    def fit(self, data, target):
        print("Training classifier")
        X, y = data, target
        self.clf.fit(X, y)
        forecast = self.clf.predict(X)
        print("prediction: ", forecast)
        print("labels:     ", y)
        i, count = 0, 0
        while i<len(y):
            if forecast[i] == y[i]:
                count += 1
            i += 1
        accuracy = count / len(y)
        print(f"Essemble Classifier Accuracy: {accuracy}")
        pass
    
    
    def predict(self, data, legal=None):
        wall = data[:4] # 0-Up, 1-Right, 2-Down, 3-Left
        food = data[4:8]  # If there is food in the respective directions
        faceghost = data[-2]
        ghost1_place = data[8:16]
        ghost2_place = data[16:24]
        escape_directions = [i for i in range(4) if wall[i] == 0] # Check for available directions without walls
        food_directions = [i for i,has_food in enumerate(food) if has_food == 1]

                
        if food_directions and faceghost == 0:
            print("Food available in directions:", food_directions)
            move = np.random.choice(food_directions)
            print("Moving towards food at direction:", move)
        
        elif 1 in ghost1_place and ghost2_place or faceghost == 1:
            print("Ghost nearby, assessing escape options.")
            # If there are legal moves not leading to a wall, choose one at random
            if escape_directions:
                print("Escape the ghost in directions:", escape_directions)
                move = np.random.choice(escape_directions)
                print("Escaping ghost to direction:", move)
            else:
            # This condition should ideally never hit 
                print("No escape available, keeping current move:",  self.previous_state)
                move = self.previous_state
        else:
            # No food directly accessible, no immediate ghost threat
            if sum(wall) <= 1:  # need to predict
                move = self.clf.predict([data])[0]
                print("Need to predict", move)
            else:
                move = self.previous_state
                print("Keep on", move)

        print("History vs. Now: ", self.previous_state, " vs ", move)
        self.previous_state = move
        return move 