
import numpy as np
import math

class ANN:
    def __init__(self):
        self.Layers = 0
        self.outputs = []
        self.weights = []
        self.Neurons = []
        self.deltas = []
        self.target = []
    
    def build(self):
        self.Layers = int(input("Enter Number of Layers: "))
        print("1 input layer, 1 output layer, and", self.Layers-2, "hidden layers are present.")
        
        for i in range(self.Layers):
            self.Neurons.append(int(input(f"Enter number of neurons in layer {i+1}: ")))
        
        for i in range(len(self.Neurons)-1):
            weights = np.random.randn(self.Neurons[i], self.Neurons[i+1]) * np.sqrt(1 / self.Neurons[i])
            self.weights.append(weights)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def Forward(self, record):
        record = np.transpose(record)
        prop = record
        self.outputs = [record]
        
        for i in range(len(self.weights)):
            move = np.dot(prop, self.weights[i])
            prop = self.sigmoid(move)
            self.outputs.append(prop)
        
        return prop
    
    def Backward(self, records, targets, learning_rate=0.1, epochs=5):
        for epoch in range(epochs):
            total_error = 0
            print(f"Epoch {epoch+1}")
            
            for idx, record in enumerate(records):
                self.target = targets[idx]
                output = self.Forward(record)
                error = self.target - output
                total_error += np.sum(error ** 2)
                
                print(f"Sample {idx+1}: Actual={output.flatten()}, Target={self.target.flatten()}, Error={error.flatten()}")
                
                self.deltas = [error * self.sigmoid_derivative(self.outputs[-1])]
                
                for i in range(len(self.weights)-1, 0, -1):
                    delta = np.dot(self.deltas[0], self.weights[i].T) * self.sigmoid_derivative(self.outputs[i])
                    self.deltas.insert(0, delta)
                
                for i in range(len(self.weights)):
                   self.weights[i] += np.dot(self.outputs[i].reshape(-1, 1), self.deltas[i].reshape(1, -1)) * learning_rate

            print(f"Total Error after Epoch {epoch+1}: {total_error}\n")
    
# Example Usage with Dataset
A = ANN()
A.build()

dataset = np.array([[0.35, 0.9], [0.45, 0.85], [0.55, 0.8]])  # 3 input samples
targets = np.array([[1], [0], [1]])  # Target labels

A.Backward(dataset, targets)  # Perform training