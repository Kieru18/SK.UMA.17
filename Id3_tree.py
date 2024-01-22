import math
import numpy as np
import pandas as pd
import random


def random_column(counts):
    max = 0
    column = 0
    for i, c in enumerate(counts):
        if c > max:
            max = c
            column = i
        elif random.getrandbits(1):
            max = c
            column = i
    return column


class Node:
    def __init__(self, attribute=None, label=None) -> None:
        self.attribute = attribute
        self.label = label
        self.children = {}  #dicitionary of children - key = attribute value, value = new nodes
    
    def add_child(self, value: float):
        self.children[int(value)] = Node()

    def __str__(self) -> str:
        return str(self.attribute)

    def display_node(self, offset):
        if(self.attribute is not None):
            print(offset*" ","(path)", self.attribute)
        else:
            print(offset*" ","(leaf)", ":", self.label)
        for child in self.children:
            print((offset+1)*" ","attribute value:", child)
            self.children[child].display_node(offset+1)


class Id3Tree:
    def __init__(self, attributes = None):
        self.root = Node()
        self.attributes = attributes
        self.classes = None
        self.dataset = None

    def entropy(self, dataset: np.ndarray) -> float:
        return -1 * sum([f * np.log(f) for f in np.unique(dataset[:, -1], return_counts=True)[1]])

    def partialEntropy(self, attribute: int, dataset: np.ndarray) -> float:
        sum = 0
        for value in np.unique(dataset[:, attribute]):
            splitData = dataset[np.where(dataset[:, attribute] == value)]
            splitEntropy = self.entropy(splitData)
            sum += len(splitData) / len(dataset) * splitEntropy
        return sum
    
    def infGain(self, attribute: int, dataset: np.ndarray) -> float:
        return self.entropy(dataset) - self.partialEntropy(attribute, dataset)
    
    def fit(self, dataset: np.ndarray):
        self.classes = np.unique(dataset[:, -1])
        self.dataset = dataset

        self.fit_recurr(self.root, self.attributes, self.dataset)

    def fit_recurr(self, node: Node, attributes: list, dataset: np.ndarray) -> None:
        classes, counts = np.unique(dataset[:, -1], return_counts=True)
        
        if len(classes) == 1:
            node.label = classes[0]
            return

        if len(attributes) == 0:
            column = random_column(counts)
            node.label = classes[column]
            return

        d = attributes[np.argmax(self.infGain(attr, dataset) for attr in attributes)]
        node.attribute = d
        for j in np.unique(dataset[:, d]):
            node.add_child(j)
            new_attributes = [attr for attr in attributes if attr != d]
            split_dataset = dataset[np.where(dataset[:, d] == j)]
            self.fit_recurr(node.children[j], new_attributes, split_dataset)
        
    
    def decide_sample(self, sample: np.ndarray):
        node = self.root  
    
        while node.attribute is not None:
            if sample[node.attribute] not in node.children.keys():
                return 0
            node = node.children[sample[node.attribute]]

        return node.label

    
    def predict(self, inputs: np.ndarray):
        return np.apply_along_axis(self.decide_sample, 1, inputs)

    def print_tree(self):
        depth = 0
        if(self.root.attribute is not None):
            print("(path):", self.root.attribute)
        else:
            print("(leaf):", self.root.label)
        for child in self.root.children:
            print(" attribute value:", child)
            self.root.children[child].display_node( depth)

        print(" ")

class RandomForest:
    def __init__(self, forest_size):
        self.trees = []
        self.forest_size = forest_size
    def train_tree(self, attributes: list, data: np.ndarray):
        tree = Id3Tree(attributes)
        tree.fit(data)
        #tree.print_tree()
        return tree
    
    def create_forest(self, dataset: pd.DataFrame):
        for i in range(1, self.forest_size+1):
            U_b = dataset.sample(n = self.forest_size, replace=True)
            k = int(math.sqrt(len( U_b.columns)))+1 # == 3
            attr_list = [i for i in range(dataset.shape[1] - 1)]
            D_b = random.sample(attr_list, k)
            self.trees.append(self.train_tree(D_b,  U_b.to_numpy()))

    
    def predict(self, dataset: np.ndarray):
        predictions = np.zeros((self.forest_size, dataset.shape[0]), dtype=int)
        predicted_values = []

        for i, tree in enumerate(self.trees):
            tmp_predict = tree.predict(dataset)
            for j in range(dataset.shape[0]):
                predictions[i][j] = tmp_predict[j]
            

        for j in range(predictions.shape[1]):
            bin_cnt = np.bincount(predictions[:, j])
            bin_cnt[0] = 0
            predicted_values.append(np.argmax(bin_cnt))
            
        return predicted_values

