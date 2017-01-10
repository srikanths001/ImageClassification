#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A simple Two Layer network with hidden RELU and a output softmax layer
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    W1,b1 and W2,b2 are weights and biases for hidden layer and output layer.
    """

    self.step_cache = {}

  def loss(self, X, model, y=None, reg=0.0):
    """
    Compute the loss and gradient.
    """
    # Unpack variables from the params dictionary
    W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']

    N, D = X.shape

    # Compute the forward pass
    scores = None
    
    #ReLU
    hidden_layer = np.maximum(0,np.dot(X,W1)+b1)
    scores = np.dot(hidden_layer,W2)+b2
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    scores -= np.max(scores)
    num = np.exp(scores)
    probs = num / np.sum(num, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(N),y])
    loss = np.sum(correct_logprobs)/N
    reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    loss += reg_loss

    # Backward pass: compute gradients
    grads = {}
    dscores = probs
    dscores[range(N),y] -= 1
    dscores /= N
    #layer 1 from end
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0)
    #hidden layer
    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer <= 0] = 0

    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0)

    dW2 += reg*W2
    dW1 += reg*W1

    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    
    return loss, grads

  def train(self, X, y, X_val, y_val, 
            model, loss_function, 
            reg=0.0,
            learning_rate=1e-2, learning_rate_decay=0.95, sample_batches=True,
            num_epochs=30, batch_size=100, acc_frequency=None,
            verbose=False):
    """
    Optimization of parameters using stochastic gradient descent with RMSprop adaptive learning rate method.
    Can supply the model params independently through 'model' dictionary.

    Returns a tuple of:
    - best_model: The model that got the highest validation accuracy during
      training.
    - loss_history: List containing the value of the loss function at each
      iteration.
    - train_acc_history: List storing the training set accuracy at each epoch.
    - val_acc_history: List storing the validation set accuracy at each epoch.
    """
    N = X.shape[0]

    if sample_batches:
      iterations_per_epoch = N / batch_size # using SGD
    else:
      iterations_per_epoch = 1 # using GD
    num_iters = num_epochs * iterations_per_epoch
    epoch = 0
    best_val_acc = 0.0
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for it in xrange(num_iters):
      if it % 10 == 0:  print 'starting iteration ', it

      # get batch of data
      if sample_batches:
        batch_mask = np.random.choice(N, batch_size)
        X_batch = X[batch_mask]
        y_batch = y[batch_mask]
      else:
        # no SGD used, full gradient descent
        X_batch = X
        y_batch = y

      # evaluate cost and gradient
      cost, grads = loss_function(X_batch, model, y_batch, reg)
      loss_history.append(cost)

      # perform a parameter update
      for p in model:
        #print("In p: %s"%p)
        # compute the parameter step
        decay_rate = 0.99 # you could also make this an option
        if not p in self.step_cache: 
          self.step_cache[p] = np.zeros(grads[p].shape)
        self.step_cache[p] = learning_rate_decay * self.step_cache[p] + (1-learning_rate_decay) * grads[p]**2
        dx = -learning_rate * grads[p]/(np.sqrt(self.step_cache[p]) + 1e-8)

        # update the parameters
        model[p] += dx

      # every epoch perform an evaluation on the validation set
      first_it = (it == 0)
      epoch_end = (it + 1) % iterations_per_epoch == 0
      acc_check = (acc_frequency is not None and it % acc_frequency == 0)
      if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
          # decay the learning rate
          learning_rate *= learning_rate_decay
          epoch += 1

        # evaluate train accuracy
        if N > 1000:
          train_mask = np.random.choice(N, 1000)
          X_train_subset = X[train_mask]
          y_train_subset = y[train_mask]
        else:
          X_train_subset = X
          y_train_subset = y
        scores_train = loss_function(X_train_subset, model)
        y_pred_train = np.argmax(scores_train, axis=1)
        train_acc = np.mean(y_pred_train == y_train_subset)
        train_acc_history.append(train_acc)

        # evaluate val accuracy
        scores_val = loss_function(X_val, model)
        y_pred_val = np.argmax(scores_val, axis=1)
        val_acc = np.mean(y_pred_val ==  y_val)
        val_acc_history.append(val_acc)
        
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
          # make a copy of the model
          best_val_acc = val_acc
          best_model = {}
          for p in model:
            best_model[p] = model[p].copy()

        # print progress if needed
        if verbose:
          print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                 % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))

    if verbose:
      print 'finished optimization. best validation accuracy: %f' % (best_val_acc, )
    # return the best model and the training history statistics
    return best_model, loss_history, train_acc_history, val_acc_history

  def predict(self, X, model=None):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    
    l1 = np.dot(X,model['W1'])+model['b1']
    hidden_lay = np.maximum(0, l1)
    scores = np.dot(hidden_lay, model['W2']) + model['b2']
    y_pred = np.argmax(scores, axis=1)

    return y_pred


