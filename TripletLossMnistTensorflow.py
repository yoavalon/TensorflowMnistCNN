import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # for data
import numpy as np

class TripletNet:
    
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 784])
        self.first = tf.placeholder(tf.float32, [None, 28, 28, 1], name='left')
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.x3 = tf.placeholder(tf.float32, [None, 784]) 
        
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.mynet(tf.reshape(self.x1,[128,28,28,1])) #self.network(self.x1)
            scope.reuse_variables() 
            self.o2 = self.mynet(tf.reshape(self.x2,[128,28,28,1])) #self.network(self.x2)
            scope.reuse_variables() 
            self.o3 = self.mynet(tf.reshape(self.x3,[128,28,28,1])) #self.network(self.x3)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.TripletLoss() #self.loss_with_spring()

    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3
    
    def mynet(self, input, reuse = tf.AUTO_REUSE) :
        
        if (reuse):
          tf.get_variable_scope().reuse_variables()
        
        with tf.name_scope("model") :          
          with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

          with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

          with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

          with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

          with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        net = tf.contrib.layers.flatten(net)
        return net
      

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def compute_euclidean_distance(self, x, y):
    
        d = tf.square(tf.sub(x, y))
        d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
        
        return d


    def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):

        with tf.name_scope("triplet_loss"):
        
          d_p_squared = tf.square(compute_euclidean_distance(anchor_feature, positive_feature))
          d_n_squared = tf.square(compute_euclidean_distance(anchor_feature, negative_feature))

          loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        #loss = d_p_squared - d_n_squared + margin

        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)

      
    #Triplet Loss 
    def TripletLoss(self) :
      
        margin = 5.0
      
        anchor_output = self.o3
        positive_output = self.o2
        negative_output = self.o1                
        
        
        with tf.name_scope("triplet_loss"):
        
          d_p_squared = tf.square(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor_output, positive_output)))))
          d_n_squared = tf.square(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor_output, negative_output)))))
          
          loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        
        #part always there
        #d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        #d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)
        
        #testing                       
        
        #losses = tf.maximum(0., margin + d_pos - d_neg, name="losses")
        #loss = tf.reduce_mean(losses, name="loss")
        #loss = 12. #tf.reduce_sum(anchor_output - positive_output)
        
        #original
        #loss = tf.maximum(0., margin + d_pos - d_neg)
        #loss = tf.reduce_mean(loss)
        
        #copied
        #losses = tf.add(pos, neg, name="losses")
        #loss = tf.reduce_mean(losses, name="loss")
        
        return tf.reduce_mean(loss)

#Get image of the opposite parity
def GetOpositeParityImage(mnist, par) :
  
  parity = par
  ran = -1
  
  while(parity == par) :
  
    ran = np.random.randint(0,mnist.train.labels.shape[0], 1)
    label = mnist.train.labels[ran]
    parity = (label % 2 == 0)
  
  return ran
      
# Create random Triplet and assign binary Label      
def GetTriplet(mnist) :
  
  ran = np.random.randint(0,mnist.train.labels.shape[0], 1)  
  lab = mnist.train.labels[ran]
  par = (lab % 2 == 0)  
  
  return np.array([ran, GetOpositeParityImage(mnist, par), GetOpositeParityImage(mnist, par), par]) #return [negative, positive, anchor, binary label]   par = 0 even, odd, odd     par = 1 odd, even, even
  
#Creates batch of shape (128,4) where as 128 is batch size and 4 stands for a triplet plus binary label
def CreateTripletBatch(mnist) :  
  Triplet_Set = []
  for i in range(128) : 
    Triplet_Set.append(GetTriplet(mnist))
  
  return np.array(Triplet_Set)

def FetchImages(mnist, indexes) : 
  
  imgList = []
  
  for i in indexes : 
    imgList.append(mnist.train.images[i])
    
  res = np.asarray(imgList)  
  
  return np.reshape(res, (128,784))

#Start 

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

g = tf.Graph() #reset graph
sess = tf.InteractiveSession(graph=g)

#Prepare Network
siamese = TripletNet();
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)

tf.initialize_all_variables().run()


for step in range(5000):
  
    Triplet = CreateTripletBatch(mnist)
          
    batch_x1 = FetchImages(mnist, Triplet[:,0])
    batch_x2 = FetchImages(mnist, Triplet[:,1])
    batch_x3 = FetchImages(mnist, Triplet[:,2])
    batch_y = np.reshape(Triplet[:,3], (128,)) 
    
    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                        siamese.x1: batch_x1,
                        siamese.x2: batch_x2,
                        siamese.x3: batch_x3,
                        siamese.y_: batch_y})
   
    if step % 1 == 0:
        print ('step %d: loss %.3f' % (step, loss_v))
