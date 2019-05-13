import numpy as np
from random import choice
from matplotlib import pyplot as plt


class GradientDescentPlot(object):

    def __init__(self, random_X=True, range_X=range(-20, 21)):
        plt.ion()
        
        # generating data
        self.X = range_X
        self.learning_rate = 0.1
        self.function = lambda x: x**2 + 1
        self.function_derivative = lambda x: 2*x
        
        # init gradient values
        self.gradient_X = choice(self.X) if random_X else np.max(self.X)
        self.gradient_Y = self.function(self.gradient_X)
    
    def show(self, loop_limit=15):
        plt.plot(self.X, [self.function(x) for x in self.X], color='black')
        plt.plot(self.X, [self.function_derivative(x) for x in self.X], color='red')
        plt.scatter([self.gradient_X], [self.gradient_Y])
    
        for i in range(loop_limit):
            self.update_plot()

    def update_plot(self):
        self.compute_gradient()
        plt.scatter([self.gradient_X], [self.gradient_Y])
        plt.pause(0.5)
        
    def compute_gradient(self):
        self.gradient_X += -self.learning_rate * self.function_derivative(self.gradient_X)
        self.gradient_Y = self.function(self.gradient_X)
        
        print('X:', self.gradient_X)
        print('Y:', self.gradient_Y)
        print()


if __name__ == '__main__':
    quadratic = GradientDescentPlot()
    quadratic.show(loop_limit=5)
    plt.clf()
