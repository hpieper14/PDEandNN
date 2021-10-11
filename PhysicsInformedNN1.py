"""
@author: Maziar Raissi
https://github.com/maziarraissi/PINNs

"""
import tensorflow as tf
import numpy as np
import time

class PINN:
    # Initialize the class
    # If we want to identify the paramters mu and nu via the NN, we set idn = True. If not, we pass in idn = False
    def __init__(self, X_u, u, X_f, layers, lb, ub, mu, nu, idn):
    
        # lower and upper bound for the spatiotemporal domain
        self.lb = lb
        self.ub = ub
    
        # training points for x and t
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        
        # values of the solution u in the training points
        self.u = u
        
        # Parameters are known
        self.nu = nu
        self.mu=mu
        
        # Intialize paramters -- will only be used when learning the parameters (is optional)
        self.mu_find = tf.Variable([0.0], dtype=tf.float32)
        self.nu_find = tf.Variable([0.0], dtype=tf.float32)
        
        # initialize tf placeholders
        self.x_u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])        
        self.u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        self.layers = layers
        
        # intialize the tensorflow session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        # initialize the collocation points if we do not want to identify the parameters
        if idn == False: 
            self.x_f = X_f[:,0:1]
            self.t_f = X_f[:,1:2]
            self.x_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
            self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
            self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        else:
            self.f_pred = self.net_f_id(self.x_u_tf, self.t_u_tf)

        # initialize u predictions via the NN      
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        # We use the L_2 loss function for both u and f 
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))
               
        
        # These are the optimizers
        # When looking at noisy data, the Adam optimizer will be used. Otherwise, it will be
        # limited memory BFGS
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        # initialize TF variables
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        
    # function that initializes the weights/biases/layers for NN. The initial weights are determined by the function xavier_init and the biases are initialized at zero.          
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
   # returns a matrix of weights sampled from truncated normal distribution                                                             
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    # Builds the neural net via $ T^k = \sigma(b + HW)$ with H defined in terms of the data matrix                                                            
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    # Defined u as a neural network                                                            
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    # Define f= u_t - N as a neural network in terms of the NN u. This is when the parameters mu and nu are known                                                            
    def net_f(self, x,t):
        mu = self.mu
        nu = self.nu
        u = self.net_u(x,t)
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxxx = tf.gradients(u_xxx, x)[0]
        f = u_t + 2*u_xx+u_xxxx-(mu - 1)*u - nu*u*u + u*u*u
        return f
    
    # Define f= u_t - N as a neural network in terms of the NN u. This is when the parameters mu and nu are NOT known                                                          
    def net_f_id(self, x,t):
        muf=self.mu_find
        nuf=self.nu_find
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxxx = tf.gradients(u_xxx, x)[0]
        f = u_t + 2*u_xx+u_xxxx-(muf - 1)*u - nuf*u*u + u*u*u
        return f
    
    # Function that prints the loss after each iteration                                                            
    def callback(self, loss):
        #print('Loss:', loss)
        None

    # Training function. Applies either the L-BFGS-B method or the adam optimizer depending on 
                                                                # whether or not the data is noisy
    def train(self, nIter, idn):
        if idn == False:
            tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                       self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        else:
            tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                mu_value = self.sess.run(self.mu_find)
                nu_value = np.exp(self.sess.run(self.nu_find))
                #print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' % 
                 #     (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    # predicts values of u using the neural network
    def predict(self, X_star, idn):
        if idn == True:
            tf_dict = {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]}
        
            u_star = self.sess.run(self.u_pred, tf_dict)
            f_star = self.sess.run(self.f_pred, tf_dict)
        else:
            u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})  
            f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
               
        return u_star, f_star