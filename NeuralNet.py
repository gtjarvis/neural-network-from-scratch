import numpy as np
import math

class neuralNet():
  def __init__(self, num_neurons, learning_rate=0.01):
    for i in range(len(num_neurons)-1):
      num_neurons[i] += 1
    self.num_inputs = num_neurons[0]
    self.num_outputs = num_neurons[len(num_neurons)-1]
    self.depth = len(num_neurons)-2
    self.num_layers = len(num_neurons)
    self.num_neurons = num_neurons
    self.weights = [np.random.random((num_neurons[i],num_neurons[i+1])) for i in range(self.num_layers-1)]
    self.learning_rate = learning_rate
    for array in self.weights:
      for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
          array[i,j] = array[i,j]*2-1
    
    
  def train(self, data, repetitions = 1000):
    print("AI Training...")
    for rep in range(repetitions):
      if rep%(int(repetitions/(min(100,repetitions)))) == 0 and not rep == 0:
        print("AI Training... (" + str(rep*100//repetitions) + "% Complete)")
        print(error)
      gradient = [np.zeros((self.num_neurons[i],self.num_neurons[i+1])) for i in range(self.num_layers-1)]
      forward_propagate_array = []
      error = 0
      for i in range(len(data)):
        forward_propagate_array = self.forward_propagate(data[i][0])
        gradient_array = self.back_propagate(forward_propagate_array, data[i][1])
        error += self.calculate_error(data[i][1],forward_propagate_array[-1])
        for j in range(len(gradient)):
          gradient[j] = np.add(gradient[j],gradient_array[j])
      for i in range(len(self.weights)):
        gradient[i] = gradient[i]*((self.learning_rate*-1)/len(data))
        self.weights[i] = np.add(self.weights[i],gradient[i])
    print("AI Training 100% Complete")
        
    
  def forward_propagate(self,inputs):
    layers = [np.zeros(self.num_neurons[i]) for i in range(self.num_layers)]
    layers[0][0] = 1
    for i in range(len(inputs)):
      layers[0][i+1] = inputs[i]
    for i in range(self.num_layers-1):
      layers[i+1] = np.matmul(layers[i],self.weights[i])
      for j in range(len(layers[i+1])):
        layers[i+1][j] = self.activation_function(layers[i+1][j])
      if i < self.num_layers-2:
        layers[i+1][0] = 1
    return layers
    
    
  def back_propagate(self, fp, target):
    gradient = [np.zeros((self.num_neurons[i],self.num_neurons[i+1])) for i in range(self.num_layers-1)]
    for i in range(len(fp[-2])):
      for j in range(len(fp[-1])):
        gradient[-1][i][j] = (fp[-1][j]-target[j])*(fp[-1][j])*(1-(fp[-1][j]))*fp[-2][i]
    for i in range(len(gradient)-2,-1,-1):
      start = 1
      if i == len(gradient)-2:
        start = 0
      for j in range(len(fp[i])):
        for k in range(len(fp[i+1])):
          sum_gradients = 0
          for l in range(start,len(fp[i+2])):
            sum_gradients += gradient[i+1][0][l]*self.weights[i+1][k][l]
          gradient[i][j][k] = (sum_gradients)*(fp[i+1][k])*(1-(fp[i+1][k]))*(fp[i][j])
    return gradient
    
  
  def derivative_error(self, target, output):
    derivative = []
    for i in range(len(target)):
      derivative.append(output[i]-target[i])
    return derivative
  
  def calculate_error(self, target, output):
    error = []
    for i in range(len(target)):
      error.append(((target[i]-output[i])**2)/2)
    return error
  
  def randomize_weights(self):
    self.weights = [np.random.random((num_neurons[i],num_neurons[i+1]))for i in range(self.num_layers-1)]
    
  def activation_function(self, x):
    return (1/(1+math.e**(-x)))
  
  def test(self, point, amax = True):
    if amax:
      answer = self.forward_propagate(point)[-1]
      greatest = 0
      decision = 0
      for j in range(len(answer)):
        if answer[j] > greatest:
          greatest = answer[j]
          decision = j
      print(decision)
      return(decision)
    else:
      print(answer)
      return answer

  def set_learning_rate(self, learning_rate):
    self.learning_rate = learning_rate

  def test_rate(self, test_points):
    percent = 0
    for i in range(len(test_points)):
      answer = self.test(test_points[i][0])
      greatest = 0
      decision = 0
      for j in range(len(answer)):
        if answer[j] > greatest:
          greatest = answer[j]
          decision = j
      if decision == test_points[i][1]:
        percent += 1
    return percent/len(test_points)

  def save_weights(self, file = "weights.txt"):
    file = open(file, "w+")
    file.write(str(self.num_layers-1) + "\n")
    for num in self.num_neurons:
      file.write(str(num) + " ")
      file.write("\n")
    for i in range(self.num_layers-1):
      for j in range(self.num_neurons[i]):
        for k in range(self.num_neurons[i+1]):
          file.write(str(self.weights[i][j][k]) + "\n")
    file.close()

  def load_weights(self, file = "weights.txt"):
    file = open(file,"r")
    lines = file.readlines()
    num_neurons = lines[1][0:-2].split(" ")
    for i in range(len(num_neurons)):
      num_neurons[i] = int(num_neurons[i])
    if num_neurons == self.num_neurons:
      c = 2
      for i in range(int(lines[0][0:-1])):
        for j in range(num_neurons[i]):
          for k in range(num_neurons[i+1]):
            self.weights[i][j][k] = float(lines[c][0:-1])
            c += 1
    else:
      print("Weights incompatible")
    file.close()
