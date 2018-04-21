import tensorflow as tf 
import matplotlib.pyplot as plt

# classification accuracy plotting
def plot_results(num_iterations=100, train_plots=[], test_plots=[]):
    colors = ['r', 'b', 'g']

    # plot training results 
    iterations = range(num_iterations)

    train = plt.figure(1)
    plt.plot(iterations, train_plots, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Train Accuracy')

    test = plt.figure(2)
    for idx in range(len(test_plots)): 
        plt.plot(iterations, test_plots[idx], colors[idx])
    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')

    plt.show() 