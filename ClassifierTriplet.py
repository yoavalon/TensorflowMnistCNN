import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


#Hyper-parameters

batchSize = 128            #128
epochs = 100              #1000
margin = 0.1               #0.1
learningRate = 0.001       #0.001
displaySteps = 1           #100
dropout = 0.5            # Dropout, probability to keep units

class TripletNet:
    
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 784])        
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.x3 = tf.placeholder(tf.float32, [None, 784]) 
        self.y_ = tf.placeholder(tf.float32, [None,2],)
                
        with tf.variable_scope("triplet") as scope:
            self.output1 = self.network(tf.reshape(self.x1,[batchSize,28,28,1])) 
            scope.reuse_variables() 
            self.output2 = self.network(tf.reshape(self.x2,[batchSize,28,28,1])) 
            scope.reuse_variables() 
            self.output3 = self.network(tf.reshape(self.x3,[batchSize,28,28,1]))             
            scope.reuse_variables() 
            self.classOutput = self.classification(self.output3)                           
        
        self.classLoss = self.ClassLoss()
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
                
        return net
      
      
    def classification(self, input, reuse = tf.AUTO_REUSE) :
        
        if (reuse):
          tf.get_variable_scope().reuse_variables()
          
        with tf.name_scope("classification") :          
        
          with tf.variable_scope("FullyConnected1") as scope:
              net = tf.contrib.layers.fully_connected(input,2,reuse=reuse,activation_fn=tf.nn.relu, scope = 'FullyConnected1')
              weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
                    
          with tf.variable_scope("FullyConnected2") as scope:
              out = tf.contrib.layers.fully_connected(net,2,reuse=reuse,activation_fn=tf.nn.relu, scope = 'FullyConnected2')
              weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)        

          with tf.variable_scope("Dropout") as scope:              
              out2 = tf.contrib.layers.dropout(out, keep_prob=dropout, noise_shape=None, is_training=True, outputs_collections=None, scope=None, seed=None)
      
        return out2
      

    def GetAccuracy(self) :  
      
        anchor_feature = self.output3
        positive_feature = self.output2
        negative_feature = self.output1    
      
        pos_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, positive_feature)),1, keepdims=True)
        neg_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, negative_feature)),1, keepdims=True)
        
        correct = tf.less_equal(pos_dis[0,:] +margin, neg_dis[0,:])
        acc = tf.reduce_sum(tf.cast(correct, tf.float32))/batchSize
                
        return acc 
      
    def TripletLoss(self) : # could replace by improved triplet loss with ratio for pos and neg. or hybrid loss function
      
        anchor_feature = self.output3
        positive_feature = self.output2
        negative_feature = self.output1          
        
        with tf.name_scope("triplet_loss"):               
          pos_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, positive_feature)),1, keepdims=True)
          neg_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, negative_feature)),1, keepdims=True)
          
          res = tf.maximum(0., pos_dis + margin - neg_dis) 
          loss = tf.reduce_mean(res)          
        
        return loss
      
    def ClassLoss(self) : #===============================================================================================================================================================
                
        with tf.variable_scope("GetClassLoss") as scope:            
            scope.reuse_variables() 
            pred = self.classification(self.output3)         
              
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = self.y_))       
        #print(loss)
           
        return loss      

    def Evaluate(self) :
      
        anchor_feature = self.output3
        positive_feature = self.output2
        negative_feature = self.output1                        
      
        with tf.name_scope("triplet_loss"):
          pos_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, positive_feature)),1, keepdims=True)
          neg_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, negative_feature)),1, keepdims=True)
          
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
  par = [(lab % 2 == 0)]
    
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

optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(model.loss)
classifier = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(model.classLoss)


tf.initialize_all_variables().run()

lossList = []
accList = []

for step in range(epochs):  
  
    TripletBatch = CreateTripletBatch(mnist)
    
    #print(TripletBatch)    
    
    batch_x1 = FetchImages(mnist, TripletBatch[:,0])
    batch_x2 = FetchImages(mnist, TripletBatch[:,1])
    batch_x3 = FetchImages(mnist, TripletBatch[:,2])
    batch_y = np.reshape(TripletBatch[:,3], (batchSize,))
    
    y_list = []
        
    
    for i in batch_y :
      if(batch_y[i]==0)  :
        y_list.append(np.array([0,1]))
      else : 
        y_list.append(np.array([1,0]))
    
    #print(y_list)
    
    _, loss_v, loss_c, Accuracy = sess.run([optimizer,  model.loss, model.classLoss ,model.Accuracy], feed_dict={
                        model.x1: batch_x1,
                        model.x2: batch_x2,
                        model.x3: batch_x3,
                        model.y_: y_list,                        
                        })    
    
    lossList.append(loss_v)
    accList.append(Accuracy)
    if step % displaySteps == 0:
        print ('step %3d:  loss: %.6f  class-loss: %.6f   triplet-accuracy: %.3f ' % (step, loss_v, loss_c, Accuracy))                        
    

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


