import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

import random
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import choice, permutation
from itertools import combinations

flags = tf.app.flags
FLAGS = flags.FLAGS

class BatchGenerator():
	def __init__(self, images, labels):
		np.random.seed(0)
		random.seed(0)
		self.labels = labels
		print images.shape
		self.images = images.reshape((55000, 28, 28, 1))
		self.tot = len(labels)
		self.i = 5
		self.num_idx = dict()
		for idx, num in enumerate(self.labels):
			if num in self.num_idx:
				self.num_idx[num].append(idx)
			else:
				self.num_idx[num] = [idx]				
		self.to_img = lambda x: self.images[x]

	def next_batch(self, batch_size):
		left = []
		right = []
		sim = []
		# genuine
		for i in range(10):
			n = 45
			l = choice(self.num_idx[i], n*2, replace=False).tolist()
			left.append(self.to_img(l.pop()))
			right.append(self.to_img(l.pop()))
			sim.append([1])
			
		#impostor
		for i,j in combinations(range(10), 2):
			left.append(self.to_img(choice(self.num_idx[i])))
			right.append(self.to_img(choice(self.num_idx[j])))
			sim.append([0])
		return np.array(left), np.array(right), np.array(sim)
		

def get_mnist():
	mnist = input_data.read_data_sets("MNIST_data/")
	return mnist


def mynet(input, reuse=False):  	
	with tf.name_scope("model"):    		
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


def contrastive_loss(model1, model2, y, margin):
	with tf.name_scope("contrastive-loss"):
		d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
		tmp= y * tf.square(d)    
		tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
		return tf.reduce_mean(tmp + tmp2) /2

mnist = get_mnist()
gen = BatchGenerator(mnist.train.images, mnist.train.labels)
test_im = np.array([im.reshape((28,28,1)) for im in mnist.test.images])
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']


left = tf.placeholder(tf.float32, [None, 28, 28, 1], name='left')
right = tf.placeholder(tf.float32, [None, 28, 28, 1], name='right')
with tf.name_scope("similarity"):
	label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	label = tf.to_float(label)
margin = 0.2


left_output = mynet(left, reuse=False)
tf.reset_default_graph() 
right_output = mynet(right, reuse=True)

loss = contrastive_loss(left_output, right_output, label, margin)

global_step = tf.Variable(0, trainable=False)


# starter_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.scalar_summary('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)


saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#setup tensorboard	
	tf.summary.scalar('step', global_step)
	tf.summary.scalar('loss', loss)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('train.log', sess.graph)

	#train iter
	for i in range(FLAGS.train_iter):
		b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)

		_, l, summary_str = sess.run([train_step, loss, merged], 
			feed_dict={left:b_l, right:b_r, label: b_sim})
		
		writer.add_summary(summary_str, i)
		print "\r#%d - Loss"%i, l

		
		if (i + 1) % FLAGS.step == 0:
			#generate test
			feat = sess.run(left_output, feed_dict={left:test_im})
			
			labels = mnist.test.labels
			# plot result
			f = plt.figure(figsize=(16,9))
			for j in range(10):
			    plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(),
			    	'.', c=c[j],alpha=0.8)
			plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
			plt.savefig('img/%d.jpg' % (i + 1))

	saver.save(sess, "model/model.ckpt")
