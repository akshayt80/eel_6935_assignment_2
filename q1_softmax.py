import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
    #reshape_done = False
    #if len(x.shape) > 1:
    #    new_dim = 1
    #    for i in x.shape:
    #        new_dim *= i
    #    x.reshape
    #axis = 1
    #if len(x.shape) == 1:
    #    axis = 0
    #    ePowx = np.exp(x - np.max(x))
    #    x = ePowx / ePowx.sum(axis=0)
    #else:
    #    ePowx = np.exp(x - np.max(x))
    #    x = ePowx / ePowx.sum(axis=0)
    #print "shape", len(x.shape)
    #scoreMatExp = np.exp(np.asarray(x))
    #x = scoreMatExp / scoreMatExp.sum(0)
    if len(x.shape) == 2:
        s = np.max(x, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(x - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        x = e_x / div
    else:
        ePowx = np.exp(x - np.max(x))
        x = ePowx / ePowx.sum()
    ### END YOUR CODE
    
    return x

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "You should verify these results!\n"

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    test1 = softmax(np.array([[1, 2, 3, 6], [2, 4, 5, 6], [1, 2, 3, 6]]))
    print test1
    assert np.max(np.fabs(test1 - np.array(
        [
            [0.00626879,  0.01704033,  0.04632042,  0.93037045], 
            [0.01203764,  0.08894681,  0.24178252,  0.657233], 
            [0.00626879,  0.01704033,  0.04632042,  0.93037045]
        ])))
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()