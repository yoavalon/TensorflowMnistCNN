import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # for data
import numpy as np

#Hyper-parameters

batchSize = 64
epochs = 20
margin = 4.
learningRate = 0.0001
#optimizer  see : http://tflearn.org/optimizers/

class TripletNet:
    
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 784])        
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.x3 = tf.placeholder(tf.float32, [None, 784]) 
        
        with tf.variable_scope("triplet") as scope:
            self.o1 = self.model(tf.reshape(self.x1,[batchSize,28,28,1])) #self.network(self.x1)
            scope.reuse_variables() 
            self.o2 = self.model(tf.reshape(self.x2,[batchSize,28,28,1])) #self.network(self.x2)
            scope.reuse_variables() 
            self.o3 = self.model(tf.reshape(self.x3,[batchSize,28,28,1])) #self.network(self.x3)
        
        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.TripletLoss() #self.loss_with_spring()
        
  
    def model(self, input, reuse = tf.AUTO_REUSE) :
        
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


    def TripletLoss(self) :
      
        anchor_output = self.o3
        positive_output = self.o2
        negative_output = self.o1                        
        
        with tf.name_scope("triplet_loss"):
        
          d_p_squared = tf.square(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor_output, positive_output)))))
          d_n_squared = tf.square(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor_output, negative_output)))))
          
          loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
      
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
  for i in range(batchSize) : 
    Triplet_Set.append(GetTriplet(mnist))
  
  return np.array(Triplet_Set)

#Fetch image data from index
def FetchImages(mnist, indexes) : 
  
  imgList = []
  
  for i in indexes : 
    imgList.append(mnist.train.images[i])
    
  res = np.asarray(imgList)  
  
  return np.reshape(res, (batchSize,784))

#Start 

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

g = tf.Graph() #reset graph
sess = tf.InteractiveSession(graph=g)

triplet = TripletNet();
#train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(triplet.loss)
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(triplet.loss)

tf.initialize_all_variables().run()

lossList = []

for step in range(epochs):  
  
    Triplet = CreateTripletBatch(mnist)
          
    batch_x1 = FetchImages(mnist, Triplet[:,0])
    batch_x2 = FetchImages(mnist, Triplet[:,1])
    batch_x3 = FetchImages(mnist, Triplet[:,2])
    batch_y = np.reshape(Triplet[:,3], (batchSize,)) 
    
    _, loss_v = sess.run([optimizer, triplet.loss], feed_dict={
                        triplet.x1: batch_x1,
                        triplet.x2: batch_x2,
                        triplet.x3: batch_x3,
                        triplet.y_: batch_y})
    
    print(a)
    
    
    lossList.append(loss_v)
    if step % 1 == 0:
        print ('step %d: loss %.3f' % (step, loss_v))

# plot Loss Graph
plt.plot(lossList)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
