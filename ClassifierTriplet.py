import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


#Hyper-parameters

batchSize = 128            #128
epochs = 100              #1000
margin = 0.1               #0.1
learningRate = 0.001       #0.001
printAfterSteps = 1      #100

class TripletNet:
    
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 784])        
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.x3 = tf.placeholder(tf.float32, [None, 784]) 
        self.y_ = tf.placeholder(tf.float32, [None])
                
        self.testX = tf.placeholder(tf.float32, [None, 784], name='testPlaceholder') 
        self.testY = tf.placeholder(tf.int32)       
        
        with tf.variable_scope("triplet") as scope:
            self.output1 = self.network(tf.reshape(self.x1,[batchSize,28,28,1])) 
            scope.reuse_variables() 
            self.output2 = self.network(tf.reshape(self.x2,[batchSize,28,28,1])) 
            scope.reuse_variables() 
            self.output3 = self.network(tf.reshape(self.x3,[batchSize,28,28,1]))             
        
        # Create loss
        
        self.loss = self.TripletLoss() 
        self.Accuracy = self.GetAccuracy()           
        
  
    def network(self, input, reuse = tf.AUTO_REUSE) :
        
        if (reuse):
          tf.get_variable_scope().reuse_variables()          
        
        with tf.name_scope("network") :          
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

        net = tf.contrib.layers.flatten(net)        #embedding
        
        return net,net
      
      
    def classification(self, input, reuse = tf.AUTO_REUSE) :
        
        if (reuse):
          tf.get_variable_scope().reuse_variables()          
        
        with tf.variable_scope("FullyConnected1") as scope:
            net = tf.contrib.layers.fully_connected(input,2,reuse=reuse,scope = 'FullyConnected1')
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        
        with tf.variable_scope("softmax") as scope:            
            net = tf.contrib.layers.softmax(net)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        
        '''
        with tf.variable_scope("FullyConnected2") as scope:
            classification = tf.contrib.layers.fully_connected(fc1,2,reuse=reuse, scope = 'FullyConnected2')
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        '''
        '''
        with tf.variable_scope("Dropout") as scope:        
            classification = tf.contrib.layers.dropout(net,keep_prob=0.5, noise_shape=None, is_training=True, outputs_collections=None, scope=None, seed=None)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        '''
        
        return net
      

    def GetAccuracy(self) :  
      
        anchor_feature = self.output3
        positive_feature = self.output2
        negative_feature = self.output1                        
      
        pos_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, positive_feature)),2, keepdims=True)
        neg_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, negative_feature)),2, keepdims=True)
        
        correct = tf.less_equal(pos_dis[0,:] +margin, neg_dis[0,:])
        acc = tf.reduce_sum(tf.cast(correct, tf.float32))/batchSize
                
        return acc 
      
    def TripletLoss(self) : # could replace by improved triplet loss with ratio for pos and neg. or hybrid loss function
      
        anchor_feature = self.output3
        positive_feature = self.output2
        negative_feature = self.output1                        
      
        with tf.name_scope("triplet_loss"):               
          pos_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, positive_feature)),2, keepdims=True)
          neg_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, negative_feature)),2, keepdims=True)
          
          res = tf.maximum(0., pos_dis + margin - neg_dis) 
          loss = tf.reduce_mean(res)
        
        return loss

    def Evaluate(self) :
      
        anchor_feature = self.output3
        positive_feature = self.output2
        negative_feature = self.output1                        
      
        with tf.name_scope("triplet_loss"):
          pos_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, positive_feature)),2, keepdims=True)
          neg_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, negative_feature)),2, keepdims=True)
          
          res = tf.maximum(0., pos_dis + margin - neg_dis) 
          loss = tf.reduce_mean(res)
        
        return loss      
      
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
  for i in range(batchSize) : 
    Triplet_Set.append(GetTriplet(mnist))
  
  return np.array(Triplet_Set)

#not in use yet
def CreateTestBatch(mnist) :  
  Test_Set = []
  for i in range(batchSize) : 
    ran = np.random.randint(0,mnist.train.labels.shape[0], 1)  
    lab = mnist.train.labels[ran]
    par = (lab % 2 == 0)  
    Test_Set.append(np.array([ran, par]))
  
  return np.array(Test_Set)

#Fetch image data from index
def FetchImages(mnist, indexes) : 
  
  imgList = []  
  for i in indexes : 
    imgList.append(mnist.train.images[i])
    
  res = np.asarray(imgList)  
  
  return np.reshape(res, (batchSize,784))

#Main

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

g = tf.Graph() #reset graph
sess = tf.InteractiveSession(graph=g)

model = TripletNet();
#optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(model.loss)
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(model.loss)
#optimizer = tf.train.RMSPropOptimizer(learning_rate = learningRate).minimize(model.loss)

tf.initialize_all_variables().run()

lossList = []
accList = []

for step in range(epochs):  
  
    TripletBatch = CreateTripletBatch(mnist)
          
    batch_x1 = FetchImages(mnist, TripletBatch[:,0])
    batch_x2 = FetchImages(mnist, TripletBatch[:,1])
    batch_x3 = FetchImages(mnist, TripletBatch[:,2])
    batch_y = np.reshape(TripletBatch[:,3], (batchSize,)) 
    
    TestBatch = CreateTestBatch(mnist)    
    test_images = FetchImages(mnist, TestBatch[:,0])
        
    _, loss_v, Accuracy = sess.run([optimizer, model.loss, model.Accuracy], feed_dict={
                        model.x1: batch_x1,
                        model.x2: batch_x2,
                        model.x3: batch_x3,
                        model.y_: batch_y,
                        model.testX : test_images,
                        model.testY : TestBatch[:,1]
                        })    
    
    lossList.append(loss_v)
    accList.append(Accuracy)
    if step % printAfterSteps == 0:
        print ('step %d: loss %.6f  training-accuracy: %.3f ' % (step, loss_v, Accuracy))                        

# plot Loss Graph
plt.plot(lossList)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# plot Loss Graph
plt.plot(accList)
plt.title('Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

#Evaluate model Accuracy on test set
model.Evaluate()


