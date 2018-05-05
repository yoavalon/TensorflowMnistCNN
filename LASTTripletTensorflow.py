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

    def TripletLoss(anchor_output, positive_output, negative_output) : #added
        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

        loss = tf.maximum(0., margin + d_pos - d_neg)
        loss = tf.reduce_mean(loss)
        
        return 1

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

def Create_Triplet_Batch(mnist) :
  
  print("create tripplet")  
  
  ranA = np.random.randint(0,mnist.train.labels.shape[0], 1)
  a_img = mnist.train.images[ranA]
  a_lab = mnist.train.labels[ranA]
  a_par = (a_lab % 2 == 0)
  
  ranB = np.random.randint(0,mnist.train.labels.shape[0], 1)
  b_img = mnist.train.images[ranB]
  b_lab = mnist.train.labels[ranB]
  b_par = (b_lab % 2 == 0)
  
  cond = True  
  ranC = 1
  
  print(a_lab)
  print(a_par)
  
  print(b_lab)
  print(b_par)
    
  while(cond) :
  
    ranC = np.random.randint(0,mnist.train.labels.shape[0], 1)
    c_lab = mnist.train.labels[ranC]
    c_par = (c_lab % 2 == 0)
    if((a_par == False and b_par == False and c_par == False) or (a_par == True and b_par == True and c_par == True))  : 
      #print("cond true")
      #print("     c_par = ", c_par)
      #print("           c_lab = ", c_lab)      
      cond = True
    else : 
      print("cond false")
      cond = False
  
  c_img = mnist.train.images[ranC]
  c_lab = mnist.train.labels[ranC]
  
  print(c_lab)
  print(c_par)
  
  print("sum")
  print(int(a_par) + int(b_par) + int(c_par))
  
  return [a_img, b_img, c_img, sum-1]
  
  
  
  #batch_x1, batch_y1 = mnist.train.next_batch(128)
  #batch_x2, batch_y2 = mnist.train.next_batch(128)
  #batch_x3, batch_y3 = mnist.train.next_batch(128)
  
  #batch_y = (batch_y1 == batch_y2).astype('float')
  

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
Create_Triplet_Batch(mnist)      

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
