from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # First reshape the input into a (N, D) matrix
    N = x.shape[0]
    D = 1
    for dim in x.shape[1:]:
        D *= dim
    
    new_x = np.reshape(x, (N, D))
    out = np.dot(new_x, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    D = 1
    for dim in x.shape[1:]:
        D *= dim

    # db is easiest: sum dout over its rows
    db = np.sum(dout, 0)
    db = db.reshape((b.shape))
    
    # for dw we first reshape x into (N, D) then do X'dout
    new_x = np.reshape(x, (N, D))
    dw = np.dot(new_x.T, dout)
    
    # for dx we do dout*w' and then reshape D to d1, ..., dk
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x, np.zeros(x.shape))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.where(cache > 0, dout, np.zeros(cache.shape))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = y.shape[0]
    exps = np.exp(x - np.max(x))
    exp_sums = np.reshape(np.sum(exps, 1), (exps.shape[0], 1))
    probs = exps / exp_sums
    loss = np.sum(-np.log(probs[range(N),y])) / N
    
    probs[range(N),y] -= 1
    dx = probs/N
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mean = np.mean(x, 0)
        var = np.var(x, 0)
        out_raw = (x - mean) / np.sqrt(var + eps)
        out = (out_raw * gamma) + beta
        
        running_mean = (momentum * running_mean) + ((1 - momentum) * mean)
        running_var = (momentum * running_var) + ((1 - momentum) * var)
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        
        cache = (x, gamma, beta, mean, var, out_raw, eps)
        
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = (x - running_mean) / np.sqrt(running_var + eps)
        out = (out * gamma) + beta
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, gamma, beta, mean, var, out_raw, eps = cache
    N, D = x.shape
    o1 = mean
    o2 = x - mean
    o3 = np.square(o2)
    o4 = np.mean(o3, 0)
    o5 = np.sqrt(o4 + eps)
    o6 = 1/o5
    o7 = o2 * o6
    
    dbeta = np.sum(dout, 0)
    dgamma = np.sum(out_raw * dout, 0)
    
    do7 = dout * gamma  # (N, D)
    do6 = np.sum(do7 * o2, 0)  # (D,)
    do2 = do7 * o6  # (N, D)
    do5 = (-1) * np.power(o5, -2) * do6
    do4 = (0.5) * do5 * np.power(o4 + eps, -0.5)
    do3 = do4 * (1/N) * np.ones((N,D))
    do2pr = do3 * 2 * o2
    do1 = (-1) * np.sum(do2 + do2pr, 0)
    do0 = do2 + do2pr
    do0pr = do1 * (1/N) * np.ones((N,D))
    dx = do0 + do0pr
    
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, gamma, beta, mean, var, out_raw, eps = cache
    N, D = x.shape
    
    mu = mean
    
    sigma = np.sqrt(var + eps)
    t1 = 1/sigma
    t2 = (1/N) * (-1 / sigma)
    
    t3_in1 = (2/N) * (x - mean)
    t3_in2 = (1/N) * (-2 / N) * np.sum(x - mean, 0)
    t3_in = t3_in1 + t3_in2
    t3_out = (x - mean) * (-1) * np.power(sigma, -2) * (0.5) * np.power(var + eps, -0.5)
    t3 = t3_out * t3_in
    
    dx_raw = t1 + t2 + t3
    dx = (gamma * dx_raw) * dout
    
    
    dbeta = np.sum(dout, 0)
    dgamma = np.sum(out_raw * dout, 0)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x = x.T
    mean = np.mean(x, 0)
    var = np.var(x, 0)
    out_raw = (x - mean) / np.sqrt(var + eps)
    out_raw = out_raw.T
    out = (out_raw * gamma) + beta
    out = out

    cache = (x, gamma, beta, mean, var, out_raw, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, gamma, beta, mean, var, out_raw, eps = cache
    
    dbeta = np.sum(dout, 0)
    dgamma = np.sum(out_raw * dout, 0)
    
    N, D = x.shape
    
    o1 = mean
    o2 = x - mean
    o3 = np.square(o2)
    o4 = np.mean(o3, 0)
    o5 = np.sqrt(o4 + eps)
    o6 = 1/o5
    o7 = o2 * o6
    
    
    
    do7 = dout * gamma
    do7 = do7.T
    do6 = np.sum(do7 * o2, 0)
    do2 = do7 * o6
    do5 = (-1) * np.power(o5, -2) * do6
    do4 = (0.5) * do5 * np.power(o4 + eps, -0.5)
    do3 = do4 * (1/N) * np.ones((N,D))
    do2pr = do3 * 2 * o2
    do1 = (-1) * np.sum(do2 + do2pr, 0)
    do0 = do2 + do2pr
    do0pr = do1 * (1/N) * np.ones((N,D))
    dx = do0 + do0pr
    dx = dx.T
    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = np.random.choice([True, False], size=x.shape, p=(p, 1-p))
        out = x * mask
        out /= p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return x, (dropout_param, mask)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask
        dx /= dropout_param["p"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    F, C, HH, WW = w.shape
    outs = []  # F elements will have shape (N, H', W')
    
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)))
    N, C, H, W = x_padded.shape
    
    for f in range(F):
        weights = w[f]  # shape (C, HH, WW)
        bias = b[f]  # int
        images = []  # N elements will have shape (H', W')
        
        for i in range(N):
            image = x_padded[i]  # shape (C, H, W)
            rows = []  # H' elements will have shape (W')
            row, col = 0, 0
            while row + HH <= H:
                cols = []  # W' ints
                while col + WW <= W:
                    cols.append(np.sum(image[:, row:row+HH, col:col+WW] * weights) + bias)
                    col += stride
                row += stride
                col = 0
                rows.append(cols)
            images.append(rows)
        
        outs.append(images)
    
    out = np.array(outs)
    out = np.swapaxes(out, 0, 1)
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # dout: (N, F, H', W')
    
    x, w, b, conv_param = cache
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    F, C, HH, WW = w.shape
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)))
    N, C, H, W = x.shape
    N, C, Hpad, Wpad = x_padded.shape
    N, F, Hprime, Wprime = dout.shape
    
    x_padded_nsummed = np.sum(x_padded, axis=0)  # (C, Hpad, Wpad)
    
    
    # dw is basically the right reshaping + sum of the x values, so we do this easily in nested loops below - finding the right x values that correspond to each change in filter value
    
    filters = np.empty((F, C, HH, WW, Hprime, Wprime))
    
    dout_nsummed = np.sum(dout, 0)  # (F, H', W')
    
    
    # Do the same process for dx as for dw - find the right filter values that will be hit by each change in x value
    
    xs = np.empty((N, C, H, W, Hprime, Wprime))
    
    for f in range(F):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for x_idx in range(0, Hpad - HH + 1, stride):
                        for y_idx in range(0, Wpad - WW + 1, stride):
                            hpr = x_idx // stride
                            wpr = y_idx // stride
                            add_to_filters = 0
                            
                            for n in range(N):
                                add_to_filters += x_padded[n, c, x_idx + hh, y_idx + ww] * dout[n, f, hpr, wpr]
                                
                            filters[f, c, hh, ww, hpr, wpr] = add_to_filters

    for n in range(N):
        for c in range(C):
            for h in range(H):
                for wi in range(W):
                    for hh in range(HH):
                        for ww in range(WW):
                            for hpr in range(Hprime):
                                for wpr in range(Wprime):
                                    # Is (Hprime, Wprime) hit by changing the x at (H, W) through filter (HH, WW)?
                                    if (h - (stride*hh) + pad == hpr) and (wi - (stride*ww) + pad == wpr):
                                        add_to_xs = 0
                                        for f in range(F):
                                            add_to_xs += w[f, c, hh, ww] * dout[n, f, hpr, wpr]
                                        xs[n, c, h, wi, hpr, wpr] = add_to_xs          
    
    dw = np.sum(filters, (4, 5))
        
    db = np.sum(dout, (0, 2, 3))
    
    dx = np.sum(xs, (4, 5))  # (N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pool_height, pool_width, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    N, C, H, W = x.shape
    
    Hprime = 1 + (H - pool_height) // stride
    Wprime = 1 + (W - pool_width) // stride
    
    out = np.empty((N, C, Hprime, Wprime))
    
    for n in range(N):
        for c in range(C):
            for h in range(0, H - pool_height + 1, stride):
                for w in range(0, W - pool_width + 1, stride):
                    curr = -np.inf
                    for dh in range(pool_height):
                        for dw in range(pool_width):
                            curr = max(curr, x[n, c, h+dh, w+dw])
                    out[n, c, h // stride, w // stride] = curr

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    pool_height, pool_width, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    N, C, H, W = x.shape
    Hprime = 1 + (H - pool_height) // stride
    Wprime = 1 + (W - pool_width) // stride
    
    dx = np.zeros((N, C, H, W))
    # dout is (N, C, Hprime, Wprime)
    
    for n in range(N):
        for c in range(C):
            for h in range(0, H - pool_height + 1, stride):
                for w in range(0, W - pool_width + 1, stride):
                    curr = -np.inf
                    curr_h = -1
                    curr_w = -1
                    for dh in range(pool_height):
                        for dw in range(pool_width):
                            if x[n, c, h+dh, w+dw] > curr:
                                curr_h = h+dh
                                curr_w = w+dw
                                curr = x[n, c, h+dh, w+dw]
                    dx[n, c, curr_h, curr_w] += dout[n, c, h // stride, w // stride]
    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    
    x_reshaped = np.swapaxes(x, 1, 3)
    x_reshaped = np.reshape(x_reshaped, (-1, x_reshaped.shape[3]))
    out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    # Reshape out back to (N, C, H, W)
    out = np.reshape(out, (N, W, H, C))
    
    out = np.swapaxes(out, 1, 3)
    
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N, C, H, W = dout.shape

    # The values in cache are still in 2D shape, from the vanilla batch norm implementation
    
    dout = np.swapaxes(dout, 1, 3)
    dout = np.reshape(dout, (-1, dout.shape[3]))
    
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    
    # Reshape dx
    dx = np.reshape(dx, (N, W, H, C))
    dx = np.swapaxes(dx, 1, 3)
                      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    dc = C // G
    
    # Call layernorm_forward() once for each G and at the end put them back together along 
    # the C dimension
    
    outs = []
    caches = []
    
    for c in range(G):
        x_split = x[:, c*dc:(c+1)*dc, :, :]
        x_split = np.reshape(x_split, (N, dc*H*W))
        
        gamma_split = gamma[:, c*dc:(c+1)*dc, :, :]
        gamma_split = np.reshape(gamma_split, (dc,))
        gamma_split = np.repeat(gamma_split, H*W)
        
        beta_split = beta[:, c*dc:(c+1)*dc, :, :]
        beta_split = np.reshape(beta_split, (dc,))
        beta_split = np.repeat(beta_split, H*W)
        

        
        out, cache = layernorm_forward(x_split, gamma_split, beta_split, gn_param)
        
        # Reshape out before appending to outs so it's easier to put back together at the end
        # Do not need to reshape cache as we will use it in its raw form in the backwards pass
        caches.append(cache)
        out = np.reshape(out, (N, dc, H, W))
        outs.append(out)
    

    
    out = np.concatenate(outs, axis=1)
    
    cache = caches

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Send each element of cache[] to layernorm_backward(), reshape them to (N, dc, H, W), then concatenate them
    
    dxs = []
    dgammas = []
    dbetas = []
    N, C, H, W = dout.shape
    G = len(cache)
    dc = C // G
    
    for idx, c in enumerate(cache):        
        dout_split = dout[:, idx*dc:(idx+1)*dc, :, :]
        dout_split = np.reshape(dout_split, (N, -1))
        
        print(dout_split.shape)
        
        dx, dgamma, dbeta = layernorm_backward(dout_split, c)

        dx = np.reshape(dx, (N, dc, H, W))
        
        dgamma = np.reshape(dgamma, (1, dc, H, W))
        dgamma = np.sum(dgamma, (2, 3))
        dgamma = np.reshape(dgamma, (1, dc, 1, 1))
        
        dbeta = np.reshape(dbeta, (1, dc, H, W))
        dbeta = np.sum(dbeta, (2, 3))
        dbeta = np.reshape(dbeta, (1, dc, 1, 1))
        
        dxs.append(dx)
        dgammas.append(dgamma)
        dbetas.append(dbeta)
    
    dx = np.concatenate(dxs, axis=1)
    dgamma = np.concatenate(dgammas, axis=1)
    dbeta = np.concatenate(dbetas, axis=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
