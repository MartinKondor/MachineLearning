import numpy as np
import random
from matplotlib import pyplot as plt


class GradientDescentPlot(object):

    def __init__(self,
                random_X=True,
                range_X=range(-20, 21),
                learning_rate=0.1,
                function=lambda x: x**2 + 1,
                function_derivative=lambda x: 2*x,
                verbose=False,
                step_delay=0.5,
                use_seaborn=True):
                
        if use_seaborn:
            import seaborn as sns
            sns.set()
            sns.set_context('paper')
        
        plt.ion()
        
        # generating data
        self.X = range_X
        self.learning_rate = learning_rate
        self.function = function
        self.function_derivative = function_derivative
        self.verbose = verbose
        self.step_delay = step_delay
        
        # init gradient values
        self.gradient_X = random.choice(self.X) if random_X else np.max(self.X)
        self.gradient_Y = self.function(self.gradient_X)
    
    def show(self, loop_limit=15, save_fig=False, save_img_name='./%d-gradient-descent-plot.png'):
        plt.plot(self.X, [self.function_derivative(x) for x in self.X], color='gray')
        plt.plot(self.X, [self.function(x) for x in self.X], color='black')
        
        # show the current gradient point
        plt.scatter([self.gradient_X], [self.gradient_Y], color='red')
    
        if save_fig:
            plt.savefig(save_img_name % 0)
    
        for i in range(loop_limit):    
            self.update_plot()
            
            if save_fig:
                plt.savefig(save_img_name % (i + 1))

    def update_plot(self):
        self.compute_gradient()
        plt.scatter([self.gradient_X], [self.gradient_Y])
        plt.pause(self.step_delay)
        
    def compute_gradient(self):
        self.gradient_X += - self.learning_rate * self.function_derivative(self.gradient_X)
        self.gradient_Y = self.function(self.gradient_X)
        
        if self.verbose:
            print('X:', self.gradient_X)
            print('Y:', self.gradient_Y)
            print()


if __name__ == '__main__':
    plt.title('Quadratic function')
    quadratic = GradientDescentPlot(
        random_X=False,
        step_delay=0.3
    ).show(loop_limit=5)
    plt.clf()
    
    plt.title('Stucking in local minima with small learning rate.')
    GradientDescentPlot(
        random_X=False,
        range_X=np.arange(-4, 5, 0.08),
        function=lambda x: x**3 + x**2 - x + 1,
        function_derivative=lambda x: 3*(x**2) + 2*x - 1,
        verbose=True,
        learning_rate=0.07
    ).show(loop_limit=7)
    plt.clf()
    
    plt.title('Going through local minima with bigger learning rate.')
    GradientDescentPlot(
        random_X=False,
        range_X=np.arange(-4, 5, 0.08),
        function=lambda x: x**3 + x**2 - x + 1,
        function_derivative=lambda x: 3*(x**2) + 2*x - 1,
        verbose=True,
        learning_rate=0.0727,
        step_delay=0.25
    ).show(loop_limit=13, save_fig=False, save_img_name='./%d-plot.png')
    
