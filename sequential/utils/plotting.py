import tensorflow as tf 
import matplotlib.pyplot as plt

# classification accuracy plotting
def plot_results(num_iterations=100, train_plots=[], test_plots=[], loss_plots=[], save=False, show=True, path='', experiment=''):
    colors = ['r', 'b', 'g']

    # plot training results 
    iterations = range(num_iterations)

    train = plt.figure()
    plt.plot(iterations, train_plots, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Train Accuracy')

    test = plt.figure()
    for idx in range(len(test_plots)): 
        plt.plot(iterations, test_plots[idx], colors[idx])
    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')

    loss = plt.figure()
    for idx in range(len(loss_plots)): 
        plt.plot(iterations, loss_plots[idx], colors[idx])
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')

    if(save):
        train.savefig(path + experiment + '_train.png')
        test.savefig(path + experiment + '_test.png')
        loss.savefig(path + experiment + '_loss.png')

    if(show):
        plt.show() 

    train.clf()
    test.clf()
    loss.clf()

def plot_varying_penalty(penalties=[], average_loss=[]):
    # plot varying loss vs. penalties results 
    loss = plt.figure() 
    plt.plot(penalties, average_loss, 'r')
    plt.xlabel('EWC Penalties')
    plt.ylabel('Average Loss')
    plt.show() 



    

