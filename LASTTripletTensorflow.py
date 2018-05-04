import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # for data
import numpy as np

class TripletNet:
    
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 784])
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.x3 = tf.placeholder(tf.float32, [None, 784]) #third element of Triplet
        
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables() 
            self.o2 = self.network(self.x2)
            scope.reuse_variables() 
            self.o3 = self.network(self.x3)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()

    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        
        C = tf.constant(margin, name="C")        
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")        
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss



      
      
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

g = tf.Graph() #reset graph
sess = tf.InteractiveSession(graph=g)

#Prepare Network
siamese = TripletNet();
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)

tf.initialize_all_variables().run()


for step in range(5000):
    batch_x1, batch_y1 = mnist.train.next_batch(128)
    batch_x2, batch_y2 = mnist.train.next_batch(128)
    batch_y = (batch_y1 == batch_y2).astype('float')
    
    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                        siamese.x1: batch_x1,
                        siamese.x2: batch_x2,
                        siamese.y_: batch_y})
   
    if step % 50 == 0:
        print ('step %d: loss %.3f' % (step, loss_v))
