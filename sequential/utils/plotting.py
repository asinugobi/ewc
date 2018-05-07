import tensorflow as tf 
import matplotlib.pyplot as plt

# classification accuracy plotting
def plot_results(num_iterations=100, train_plots=[], test_plots=[], loss_plots=[], save=False, show=True, path='', experiment='', title=''):
    colors = ['r', 'b', 'g']
    labels = ['Source', 'Target', 'Target 2'] 

    # plot training results 
    iterations = range(num_iterations)

    train = plt.figure()
    plt.plot(iterations, train_plots, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Train Accuracy')
    plt.title(title + ': Train Accuracy vs. Iterations')

    test = plt.figure()
    for idx in range(len(test_plots)): 
        plt.plot(iterations, test_plots[idx], colors[idx], label=labels[idx])
    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')
    plt.title(title + ': Test Accuracy vs. Iterations')
    plt.legend(labels[:len(test_plots)])

    loss = plt.figure()
    for idx in range(len(loss_plots)): 
        plt.plot(iterations, loss_plots[idx], colors[idx], label=labels[idx])
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.title(title + ': Average Loss vs. Iterations')
    plt.legend(labels[:len(loss_plots)])

    if(save):
        train.savefig(path + experiment + '_train.png')
        test.savefig(path + experiment + '_test.png')
        loss.savefig(path + experiment + '_loss.png')

    if(show):
        plt.show() 

    plt.close()
    plt.close()
    plt.close()

def plot_varying_penalty(penalties=[], average_loss=[], path='', experiment='', save=False, show=False, title=''):
    # plot varying loss vs. penalties results 
    loss = plt.figure() 
    plt.plot(penalties, average_loss, 'r.')
    plt.xlabel('EWC Penalty')
    plt.ylabel('Average Loss')
    plt.title(title + ': Average Loss vs. EWC Penalty') 

    if (save): 
        loss.savefig(path + experiment + 'loss_vs_penalty.png')
    
    if (show): 
        plt.show() 
    
    # close plot 
    plt.close() 



    

