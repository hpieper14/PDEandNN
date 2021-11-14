#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
https://github.com/maziarraissi/DeepHPMs

Modified by Hannah Pieper for the Swift-Hohenberg Equation 

"""
import tensorflow as tf
import numpy as np
import time

### Note that these three functions are the same as in the PINN class
# initialize the NN 
def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases

# used to initialize the weights in the NN
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)

# constructs NN. Note that the activation function is sinusoidal now
def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


class DeepHPM:
    # initialize DeepHPM class attributes
    def __init__(self, t, x, u,
                       x0, u0, tb, X_f,
                       u_layers, pde_layers,
                       layers,
                       lb_idn, ub_idn,
                       lb_sol, ub_sol):
        
        # Domain Boundaries
        self.lb_idn = lb_idn
        self.ub_idn = ub_idn
        self.lb_sol = lb_sol
        self.ub_sol = ub_sol
        
        # Initialize for Identification of N
        self.idn_init(t, x, u, u_layers, pde_layers)
        
        # Initialize for Solution u
        self.sol_init(x0, u0, tb, X_f, layers)
        
        # tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    ###########################################################################
    ############################# Identifier ##################################
    ###########################################################################
    # initialize the variables for identifying N   
    def idn_init(self, t, x, u, u_layers, pde_layers):
        # Training Data for Identification
        self.t = t
        self.x = x
        self.u = u
        
        # Layers for identification
        self.u_layers = u_layers
        self.pde_layers = pde_layers
        
        # Initialize NNs for identification
        self.u_weights, self.u_biases = initialize_NN(u_layers)
        self.pde_weights, self.pde_biases = initialize_NN(pde_layers)
        
        # tf placeholders for identification
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.terms_tf = tf.placeholder(tf.float32, shape=[None, pde_layers[0]])
        
        # tf graphs for identification
        self.idn_u_pred = self.idn_net_u(self.t_tf, self.x_tf)
        self.pde_pred = self.net_pde(self.terms_tf)
        self.idn_f_pred = self.idn_net_f(self.t_tf, self.x_tf)
        
        # loss for identification
        self.idn_u_loss = tf.reduce_sum(tf.square(self.idn_u_pred - self.u_tf))
        self.idn_f_loss = tf.reduce_sum(tf.square(self.idn_f_pred))
        
        # Optimizer for identification
        self.idn_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_u_loss,
                               var_list = self.u_weights + self.u_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 50000,
                                          'maxfun': 50000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})
    
        self.idn_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_f_loss,
                               var_list = self.pde_weights + self.pde_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 50000,
                                          'maxfun': 50000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})
    
      

    
        self.idn_u_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_u_train_op_Adam = self.idn_u_optimizer_Adam.minimize(self.idn_u_loss, 
                                   var_list = self.u_weights + self.u_biases)
        
        self.idn_f_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_f_train_op_Adam = self.idn_f_optimizer_Adam.minimize(self.idn_f_loss, 
                                   var_list = self.pde_weights + self.pde_biases)  
    # construct the neural network for N
    def idn_net_u(self, t, x):
        X = tf.concat([t,x],1)
        H = 2.0*(X - self.lb_idn)/(self.ub_idn - self.lb_idn) - 1.0
        u = neural_net(H, self.u_weights, self.u_biases)
        return u
    # construct the neural network for u
    def net_pde(self, terms):
        pde = neural_net(terms, self.pde_weights, self.pde_biases)
        return pde
    
    # construct the neural network for f = u_t - N
    def idn_net_f(self, t, x):
        u = self.idn_net_u(t, x)
        
        u_t = tf.gradients(u, t)[0]
        
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxxx = tf.gradients(u_xxx, x)[0]
        
        terms = tf.concat([u,u_x,u_xx, u_xxx, u_xxxx],1)

        
        f = u_t - self.net_pde(terms)
        
        return f

    # train the neural network for u
    def idn_u_train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.u_tf: self.u}
        
        start_time = time.time()
        for it in range(N_iter):
            
            self.sess.run(self.idn_u_train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_u_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        self.idn_u_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_u_loss],
                                      loss_callback = self.callback)
    # train the neural network for f
    def idn_f_train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x}
        
        start_time = time.time()
        for it in range(N_iter):
            
            self.sess.run(self.idn_f_train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_f_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        self.idn_f_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_f_loss],
                                      loss_callback = self.callback)
    # predict using the models generated for f and u
    def idn_predict(self, t_star, x_star):
        
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star}
        
        u_star = self.sess.run(self.idn_u_pred, tf_dict)
        f_star = self.sess.run(self.idn_f_pred, tf_dict)
        
        return u_star, f_star
    # predict the pde
    def predict_pde(self, terms_star):
        
        tf_dict = {self.terms_tf: terms_star}
        
        pde_star = self.sess.run(self.pde_pred, tf_dict)
        
        return pde_star
    
    ###########################################################################
    ############################### Solver ####################################
    ###########################################################################
    
    # this portion of the code is designed to work with the models constructed in the previous portion.
    # The goal to construct a solution u(t,x) to u_t = N, generated above
    
    # intialize the necessary variables
    def sol_init(self, x0, u0, tb, X_f, layers):
        # Training Data for Solution
        X0 = np.concatenate((0*x0, x0), 1) # (0, x0)
        # lower and upper bounds for the domain
        X_lb = np.concatenate((tb, 0*tb + self.lb_sol[1]), 1) # (tb, lb[1])
        X_ub = np.concatenate((tb, 0*tb + self.ub_sol[1]), 1) # (tb, ub[1])
                
        self.X_f = X_f # Collocation Points
        self.t0 = X0[:,0:1] # Initial Data (time)
        self.x0 = X0[:,1:2] # Initial Data (space)
        self.t_lb = X_lb[:,0:1] # Boundary Data (time) -- lower boundary
        self.x_lb = X_lb[:,1:2] # Boundary Data (space) -- lower boundary
        self.t_ub = X_ub[:,0:1] # Boundary Data (time) -- upper boundary
        self.x_ub = X_ub[:,1:2] # Boundary Data (space) -- upper boundary
        self.t_f = X_f[:,0:1] # Collocation Points (time)
        self.x_f = X_f[:,1:2] # Collocation Points (space)
        self.u0 = u0 # Boundary Data
        
        # Layers for Solution
        self.layers = layers
        
        # Initialize NNs for Solution
        self.weights, self.biases = initialize_NN(layers)
        
        # tf placeholders for Solution
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # tf graphs for Solution
        self.u0_pred, _  = self.sol_net_u(self.t0_tf, self.x0_tf)
        self.u_lb_pred, self.u_x_lb_pred = self.sol_net_u(self.t_lb_tf, self.x_lb_tf)
        self.u_ub_pred, self.u_x_ub_pred = self.sol_net_u(self.t_ub_tf, self.x_ub_tf)
        self.sol_f_pred = self.sol_net_f(self.t_f_tf, self.x_f_tf)
        
        # loss for Solution
        self.sol_loss = tf.reduce_sum(tf.square(self.u0_tf - self.u0_pred)) + \
                        tf.reduce_sum(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                        tf.reduce_sum(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                        tf.reduce_sum(tf.square(self.sol_f_pred))
        
        # Optimizer for Solution
        self.sol_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.sol_loss,
                             var_list = self.weights + self.biases,
                             method = 'L-BFGS-B',
                             options = {'maxiter': 50000,
                                        'maxfun': 50000,
                                        'maxcor': 50,
                                        'maxls': 50,
                                        'ftol': 1.0*np.finfo(float).eps})
    
    
        self.sol_optimizer_Adam = tf.train.AdamOptimizer()
        self.sol_train_op_Adam = self.sol_optimizer_Adam.minimize(self.sol_loss,
                                 var_list = self.weights + self.biases)
    # build the NN for u, the solution
    def sol_net_u(self, t, x):
        X = tf.concat([t,x],1)
        H = 2.0*(X - self.lb_sol)/(self.ub_sol - self.lb_sol) - 1.0
        u = neural_net(H, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        return u, u_x
    
    # build the NN for f = u_t - N (now we know N from the above work)
    def sol_net_f(self, t, x):
        u, _ = self.sol_net_u(t,x)
        
        u_t = tf.gradients(u, t)[0]
        
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxxx = tf.gradients(u_xxx, x)[0]
        
        terms = tf.concat([u,u_x,u_xx, u_xxx, u_xxxx],1)
        
        f = u_t - self.net_pde(terms)
        
        return f
    # used to print the loss on each iteration
    def callback(self, loss):
        #print('Loss: %e' % (loss))
        None
    
    # train the solution NN u
    def sol_train(self, N_iter):
        tf_dict = {self.t0_tf: self.t0, self.x0_tf: self.x0,
                   self.u0_tf: self.u0,
                   self.t_lb_tf: self.t_lb, self.x_lb_tf: self.x_lb,
                   self.t_ub_tf: self.t_ub, self.x_ub_tf: self.x_ub,
                   self.t_f_tf: self.t_f, self.x_f_tf: self.x_f}
        
        start_time = time.time()
        for it in range(N_iter):
            
            self.sess.run(self.sol_train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.sol_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                
        self.sol_optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.sol_loss],
                                    loss_callback = self.callback)
    # predict values on (t,x) using the NN u
    def sol_predict(self, t_star, x_star):
        
        u_star = self.sess.run(self.u0_pred, {self.t0_tf: t_star, self.x0_tf: x_star})  
        f_star = self.sess.run(self.sol_f_pred, {self.t_f_tf: t_star, self.x_f_tf: x_star})
               
        return u_star, f_star    

