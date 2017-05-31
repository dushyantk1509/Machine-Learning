import pickle
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

train_data1 = unpickle("data_batch_1")
train_data2 = unpickle("data_batch_2")
train_data3 = unpickle("data_batch_3")
train_data4 = unpickle("data_batch_4")
train_data5 = unpickle("data_batch_5")
test_data = unpickle("test_batch")

def one_hot_encode(y,n_classes=10):
	return np.eye(n_classes)[y]

def normalize_columns(arr):
	rows,cols = np.shape(arr)
	for col in range(cols):
		arr[:,col] = (arr[:,col] - np.mean(arr[:,col]))/(abs(arr[:,col]).max() - abs(arr[:,col]).min())

#print(train_data1[b'data'])
train_data1_xs = train_data1[b'data'];
train_data1_xs = np.array(train_data1_xs,dtype=np.float64)
normalize_columns(train_data1_xs);
#print(train_data1_xs)
#print(np.mean(train_data1_xs[:,1]))
train_data1_ys = train_data1[b'labels'];
train_data1_ys = one_hot_encode(train_data1_ys);
#print(train_data1_ys)

train_data2_xs = train_data2[b'data'];
train_data2_xs = np.array(train_data2_xs,dtype=np.float64)
normalize_columns(train_data2_xs);
train_data2_ys = train_data2[b'labels'];
train_data2_ys = one_hot_encode(train_data2_ys);

train_data3_xs = train_data3[b'data'];
train_data3_xs = np.array(train_data3_xs,dtype=np.float64)
normalize_columns(train_data3_xs);
train_data3_ys = train_data3[b'labels'];
train_data3_ys = one_hot_encode(train_data3_ys);

train_data4_xs = train_data4[b'data'];
train_data4_xs = np.array(train_data4_xs,dtype=np.float64)
normalize_columns(train_data4_xs);
train_data4_ys = train_data4[b'labels'];
train_data4_ys = one_hot_encode(train_data4_ys);

train_data5_xs = train_data5[b'data'];
train_data5_xs = np.array(train_data5_xs,dtype=np.float64)
normalize_columns(train_data5_xs);
train_data5_ys = train_data5[b'labels'];
train_data5_ys = one_hot_encode(train_data5_ys);

test_data_xs = test_data[b'data'];
test_data_xs = np.array(test_data_xs,dtype=np.float64)
normalize_columns(test_data_xs);
test_data_ys = test_data[b'labels'];
test_data_ys = one_hot_encode(test_data_ys);

x = tf.placeholder(tf.float32, shape=[None, 3072,])
y_labels = tf.placeholder(tf.float32, shape=[None, 10])

img_data = tf.reshape(x, [-1,3,32,32])
x_image = tf.transpose(img_data, perm=[0,2,3,1])
global_step = tf.Variable(0,trainable=False)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

kernel1 = weight_variable([3, 3, 3, 32])
conv1 = tf.nn.conv2d(x_image,kernel1,[1,1,1,1],padding='SAME')
bias1 = bias_variable([32])
pre_activation1 = tf.nn.bias_add(conv1,bias1)
conv1_pre_final = tf.nn.relu(pre_activation1)
batch_mean1, batch_var1 = tf.nn.moments(conv1_pre_final,[0,1,2])
beta = tf.Variable(tf.constant(0.0,shape=[32]))
gamma = tf.Variable(tf.constant(1.0,shape=[32]))
conv1_final = tf.nn.batch_normalization(conv1_pre_final,batch_mean1,batch_var1,beta,gamma,1e-3)
#pool1 = tf.nn.max_pool(conv1_final,ksize = [1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
#norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

kernel2 = weight_variable([3, 3, 32, 32])
conv2 = tf.nn.conv2d(conv1_final,kernel2,[1,1,1,1],padding='SAME')
bias2 = bias_variable([32])
pre_activation2 = tf.nn.bias_add(conv2,bias2)
conv2_pre_final = tf.nn.relu(pre_activation2)
batch_mean2, batch_var2 = tf.nn.moments(conv2_pre_final,[0,1,2])
#beta = tf.Variable(tf.constant(0.0,shape=[32]))
#gamma = tf.Variable(tf.constant(1.0,shape=[32]))
conv2_final = tf.nn.batch_normalization(conv2_pre_final,batch_mean2,batch_var2,beta,gamma,1e-3)
pool1 = tf.nn.max_pool(conv2_final,ksize = [1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
pool1_drop = tf.nn.dropout(pool1,0.2)

kernel3 = weight_variable([3, 3, 32, 64])
conv3 = tf.nn.conv2d(pool1_drop,kernel3,[1,1,1,1],padding='SAME')
bias3 = bias_variable([64])
pre_activation3 = tf.nn.bias_add(conv3,bias3)
conv3_pre_final = tf.nn.relu(pre_activation3)
batch_mean3, batch_var3 = tf.nn.moments(conv3_pre_final,[0,1,2])
beta = tf.Variable(tf.constant(0.0,shape=[64]))
gamma = tf.Variable(tf.constant(1.0,shape=[64]))
conv3_final = tf.nn.batch_normalization(conv3_pre_final,batch_mean3,batch_var3,beta,gamma,1e-3)
#pool1 = tf.nn.max_pool(conv1_final,ksize = [1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
#norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

kernel4 = weight_variable([3, 3, 64, 64])
conv4 = tf.nn.conv2d(conv3_final,kernel4,[1,1,1,1],padding='SAME')
bias4 = bias_variable([64])
pre_activation4 = tf.nn.bias_add(conv4,bias4)
conv4_pre_final = tf.nn.relu(pre_activation4)
batch_mean4, batch_var4 = tf.nn.moments(conv4_pre_final,[0,1,2])
#beta = tf.Variable(tf.constant(0.0,shape=[64]))
#gamma = tf.Variable(tf.constant(1.0,shape=[64]))
conv4_final = tf.nn.batch_normalization(conv4_pre_final,batch_mean4,batch_var4,beta,gamma,1e-3)
pool4 = tf.nn.max_pool(conv4_final,ksize = [1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
pool4_drop = tf.nn.dropout(pool4,0.2)

kernel5 = weight_variable([5, 5, 64, 128])
conv5 = tf.nn.conv2d(pool4_drop,kernel5,[1,1,1,1],padding='SAME')
bias5 = bias_variable([128])
pre_activation5 = tf.nn.bias_add(conv5,bias5)
conv5_pre_final = tf.nn.relu(pre_activation5)
batch_mean5, batch_var5 = tf.nn.moments(conv5_pre_final,[0,1,2])
beta = tf.Variable(tf.constant(0.0,shape=[128]))
gamma = tf.Variable(tf.constant(1.0,shape=[128]))
conv5_final = tf.nn.batch_normalization(conv5_pre_final,batch_mean5,batch_var5,beta,gamma,1e-3)
pool5 = tf.nn.max_pool(conv5_final,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
pool5_drop = tf.nn.dropout(pool5,0.2)

W_fc1 = weight_variable([4 * 4 * 128, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(pool5_drop, [-1, 4*4*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

W_fc2 = weight_variable([256, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y_conv))
train_step = tf.train.MomentumOptimizer(0.01, momentum=0.9).minimize(cross_entropy,global_step)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for j in range(1000000):
	k = j%500
	if k<100:
		i = k
		batch_xs = train_data1_xs[100*i:100*(i+1)]
		batch_ys = train_data1_ys[100*i:100*(i+1)]
	elif 200>k>=100:
		i = k-100
		batch_xs = train_data2_xs[100*i:100*(i+1)]
		batch_ys = train_data2_ys[100*i:100*(i+1)]
	elif 300>k>=200:
		i = k-200
		batch_xs = train_data3_xs[100*i:100*(i+1)]
		batch_ys = train_data3_ys[100*i:100*(i+1)]
	elif 400>k>=300:
		i = k-300
		batch_xs = train_data4_xs[100*i:100*(i+1)]
		batch_ys = train_data4_ys[100*i:100*(i+1)]
	else:
		i = k-400
		batch_xs = train_data5_xs[100*i:100*(i+1)]
		batch_ys = train_data5_ys[100*i:100*(i+1)]

	if j%25 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_labels:batch_ys})
		print("step %d, training accuracy %g"%(j, train_accuracy))

	if j%100 == 0:
		test_acc = 0.0
		for i in range(100):
			test_xs = test_data_xs[100*i:100*(i+1)]
			test_ys = test_data_ys[100*i:100*(i+1)]
			acc = accuracy.eval(feed_dict={x:test_xs,y_labels:test_ys})
			test_acc += acc

		print("training accuracy on test_data : %g"%(test_acc/100.0))

	train_step.run(feed_dict={x:batch_xs, y_labels:batch_ys})

test_acc = 0.0

for i in range(100):
	test_xs = test_data[b'data'][100*i:100*(i+1)]
	batch_xs = np.array(batch_xs, dtype=np.float32)
	normalize_columns(batch_xs)
	test_ys = test_data[b'labels'][100*i:100*(i+1)]
	test_ys = list_connversion(test_ys)

	acc = accuracy.eval(feed_dict={x:test_xs,y_:test_ys,keep_prob:1.0})
	test_acc += acc
	print(acc)
	print(test_acc)

print("training accuracy %g"%(test_acc/100.0))
