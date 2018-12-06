import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class CNN(Chain):

	def __init__(self):
		super(CNN, self).__init__()
		with self.init_scope():
			# Convolution
			self.conv1 = L.Convolution2D(None, 32, (3,3), stride=1, pad=1)	#64->64
			self.conv2 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)	#32->32
			self.conv3 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#16->16
			self.conv4 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#8->8
			self.conv5 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv6 = L.Convolution2D(None, 512, (2,2), stride=1)		#2->1

			# BatchNormalize
			self.bn1 = L.BatchNormalization(32)
			self.bn2 = L.BatchNormalization(64)
			self.bn3 = L.BatchNormalization(128)
			self.bn4 = L.BatchNormalization(256)
			self.bn5 = L.BatchNormalization(512)
			self.bn6 = L.BatchNormalization(512)

			self.fc = L.Linear(None, 2)

	def __call__(self, x):
		# CNN
		self.h_conv1 = F.relu(self.bn1(self.conv1(x)))
		self.h_pool1 = F.max_pooling_2d(self.h_conv1, (2,2), stride=2)	#64->32
		self.h_conv2 = F.relu(self.bn2(self.conv2(self.h_pool1)))
		self.h_pool2 = F.max_pooling_2d(self.h_conv2, (2,2), stride=2)	#32->16
		self.h_conv3 = F.relu(self.bn3(self.conv3(self.h_pool2)))
		self.h_pool3 = F.max_pooling_2d(self.h_conv3, (2,2), stride=2)	#16->8
		self.h_conv4 = F.relu(self.bn4(self.conv4(self.h_pool3)))
		self.h_pool4 = F.max_pooling_2d(self.h_conv4, (2,2), stride=2)	#8->4
		self.h_conv5 = F.relu(self.bn5(self.conv5(self.h_pool4)))
		self.h_pool5 = F.max_pooling_2d(self.h_conv5, (2,2), stride=2)	#4->2
		self.h_conv6 = F.relu(self.bn6(self.conv6(self.h_pool5)))

		# fc
		self.y = self.fc(F.dropout(self.h_conv6.reshape(-1, 512), ratio=.9))

		return self.y


class CNN2(Chain):

	def __init__(self):
		super(CNN2, self).__init__()
		with self.init_scope():
			# Convolution
			self.conv1 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)	#64->64
			self.conv2 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#32->32
			self.conv3 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv4 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#8->8
			self.conv5 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv6 = L.Convolution2D(None, 512, (2,2), stride=1)	#2->1

			# BatchNormalize
			self.bn1 = L.BatchNormalization(64)
			self.bn2 = L.BatchNormalization(128)
			self.bn3 = L.BatchNormalization(256)
			self.bn4 = L.BatchNormalization(256)
			self.bn5 = L.BatchNormalization(512)
			self.bn6 = L.BatchNormalization(512)

			self.fc = L.Linear(None, 2)

	def __call__(self, x):
		# CNN
		self.h_conv1 = F.tanh(self.bn1(self.conv1(x)))
		self.h_pool1 = F.max_pooling_2d(self.h_conv1, (2,2), stride=2)	#64->32
		self.h_conv2 = F.tanh(self.bn2(self.conv2(self.h_pool1)))
		self.h_pool2 = F.max_pooling_2d(self.h_conv2, (2,2), stride=2)	#32->16
		self.h_conv3 = F.tanh(self.bn3(self.conv3(self.h_pool2)))
		self.h_pool3 = F.max_pooling_2d(self.h_conv3, (2,2), stride=2)	#16->8
		self.h_conv4 = F.tanh(self.bn4(self.conv4(self.h_pool3)))
		self.h_pool4 = F.max_pooling_2d(self.h_conv4, (2,2), stride=2)	#8->4
		self.h_conv5 = F.tanh(self.bn5(self.conv5(self.h_pool4)))
		self.h_pool5 = F.max_pooling_2d(self.h_conv5, (2,2), stride=2)	#4->2
		self.h_conv6 = F.tanh(self.bn6(self.conv6(self.h_pool5)))

		# fc
		self.y = self.fc(F.dropout(self.h_conv6.reshape(-1, 512), ratio=.9))

		return self.y

class CNN3(Chain):

	def __init__(self):
		super(CNN3, self).__init__()
		with self.init_scope():
			# Convolution
			self.conv1_1 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)	#64->64
			self.conv2_1 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#32->32
			self.conv3_1 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv4_1 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#8->8
			self.conv5_1 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv6_1 = L.Convolution2D(None, 512, (2,2), stride=1)	#2->1

			self.conv1_2 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)	#64->64
			self.conv2_2 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#32->32
			self.conv3_2 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv4_2 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#8->8
			self.conv5_2 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv6_2 = L.Convolution2D(None, 512, (2,2), stride=1)	#2->1

			# BatchNormalize
			self.bn1_1 = L.BatchNormalization(64)
			self.bn2_1 = L.BatchNormalization(128)
			self.bn3_1 = L.BatchNormalization(256)
			self.bn4_1 = L.BatchNormalization(256)
			self.bn5_1 = L.BatchNormalization(512)
			self.bn6_1 = L.BatchNormalization(512)

			self.bn1_2 = L.BatchNormalization(64)
			self.bn2_2 = L.BatchNormalization(128)
			self.bn3_2 = L.BatchNormalization(256)
			self.bn4_2 = L.BatchNormalization(256)
			self.bn5_2 = L.BatchNormalization(512)
			self.bn6_2 = L.BatchNormalization(512)

			self.fc = L.Linear(None, 2)

	def __call__(self, x):
		# CNN
		self.h_conv1_1 = F.tanh(self.bn1_1(self.conv1_1(x)))
		self.h_conv1_2 = F.relu(self.bn1_2(self.conv1_2(x)))
		self.concat1 = F.concat((self.h_conv1_1, self.h_conv1_2), axis=1)
		self.h_pool1 = F.max_pooling_2d(self.concat1, (2,2), stride=2)	#64->32

		self.h_conv2_1 = F.tanh(self.bn2_1(self.conv2_1(self.h_pool1)))
		self.h_conv2_2 = F.relu(self.bn2_2(self.conv2_2(self.h_pool1)))
		self.concat2 = F.concat((self.h_conv2_1, self.h_conv2_2), axis=1)
		self.h_pool2 = F.max_pooling_2d(self.concat2, (2,2), stride=2)	#32->16

		self.h_conv3_1 = F.tanh(self.bn3_1(self.conv3_1(self.h_pool2)))
		self.h_conv3_2 = F.relu(self.bn3_2(self.conv3_2(self.h_pool2)))
		self.concat3 = F.concat((self.h_conv3_1, self.h_conv3_2), axis=1)
		self.h_pool3 = F.max_pooling_2d(self.concat3, (2,2), stride=2)	#16->8

		self.h_conv4_1 = F.tanh(self.bn4_1(self.conv4_1(self.h_pool3)))
		self.h_conv4_2 = F.relu(self.bn4_2(self.conv4_2(self.h_pool3)))
		self.concat4 = F.concat((self.h_conv4_1, self.h_conv4_2), axis=1)
		self.h_pool4 = F.max_pooling_2d(self.concat4, (2,2), stride=2)	#8->4

		self.h_conv5_1 = F.tanh(self.bn5_1(self.conv5_1(self.h_pool4)))
		self.h_conv5_2 = F.relu(self.bn5_2(self.conv5_2(self.h_pool4)))
		self.concat5 = F.concat((self.h_conv5_1, self.h_conv5_2), axis=1)
		self.h_pool5 = F.max_pooling_2d(self.concat5, (2,2), stride=2)	#4->2

		self.h_conv6_1 = F.tanh(self.bn6_1(self.conv6_1(self.h_pool5)))
		self.h_conv6_2 = F.relu(self.bn6_2(self.conv6_2(self.h_pool5)))
		self.concat6 = F.concat((self.h_conv6_1, self.h_conv6_2), axis=1)

		# fc
		self.y = self.fc(F.dropout(self.concat6.reshape(-1, 512*2), ratio=.9))

		return self.y


class CNN_ReNorm(Chain):

	def __init__(self):
		super(CNN_ReNorm, self).__init__()
		with self.init_scope():
			# Convolution
			self.conv1_1 = L.Convolution2D(None, 32, (5,5), stride=1, pad=2)	#64->64
			self.conv1_2 = L.Convolution2D(None, 32, (5,5), stride=1, pad=2)	#64->64
			self.conv2_1 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)	#32->32
			self.conv2_2 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)	#32->32
			self.conv3_1 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#16->16
			self.conv3_2 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#16->16
			self.conv4_1 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#8->8
			self.conv4_2 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#8->8
			self.conv5_1 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv5_2 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv6 = L.Convolution2D(None, 512, (2,2), stride=1)		#2->1

			# BatchNormalize
			self.bn1 = L.BatchNormalization(32)
			self.bn2 = L.BatchNormalization(64)
			self.bn3 = L.BatchNormalization(128)
			self.bn4 = L.BatchNormalization(256)
			self.bn5 = L.BatchNormalization(512)
			self.bn6 = L.BatchNormalization(512)

			self.fc1 = L.Linear(None, 256)
			self.fc2 = L.Linear(None, 256)
			self.fc3 = L.Linear(None, 2)

	def __call__(self, x):
		# CNN
		self.h_conv1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(x)))))
		self.h_pool1 = F.max_pooling_2d(self.h_conv1, (2,2), stride=2)	#64->32
		self.h_conv2 = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(self.h_pool1)))))
		self.h_pool2 = F.max_pooling_2d(self.h_conv2, (2,2), stride=2)	#32->16
		self.h_conv3 = F.relu(self.bn3(self.conv3_2(F.relu(self.conv3_1(self.h_pool2)))))
		self.h_pool3 = F.max_pooling_2d(self.h_conv3, (2,2), stride=2)	#16->8
		self.h_conv4 = F.relu(self.bn4(self.conv4_2(F.relu(self.conv4_1(self.h_pool3)))))
		self.h_pool4 = F.max_pooling_2d(self.h_conv4, (2,2), stride=2)	#8->4
		self.h_conv5 = F.relu(self.bn5(self.conv5_2(F.relu(self.conv5_1(self.h_pool4)))))
		self.h_pool5 = F.max_pooling_2d(self.h_conv5, (2,2), stride=2)	#4->2
		self.h_conv6 = F.relu(self.bn6(self.conv6(self.h_pool5)))	#2->1

		# fc
		self.fc_1 = F.relu(self.fc1(F.dropout(self.h_conv6.reshape(-1, 512), ratio=.5)))
		self.fc_2 = F.relu(F.dropout(self.fc_1, ratio=.5))
		self.y = self.fc3(F.dropout(self.fc_2, ratio=.5))

		return self.y


class VGG(Chain):

	def __init__(self):
		super(VGG, self).__init__()
		with self.init_scope():
			# Convolution
			self.conv1_1 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)    #64->64
			self.conv1_2 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)    #64->64
			self.conv2_1 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#32->32
			self.conv2_2 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#32->32
			self.conv3_1 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv3_2 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv3_3 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv4_1 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#8->8
			self.conv4_2 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#8->8
			self.conv4_3 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#8->8
			self.conv5_1 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv5_2 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv5_3 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4

			# BatchNormalize
			self.bn1 = L.BatchNormalization(64)
			self.bn2 = L.BatchNormalization(128)
			self.bn3 = L.BatchNormalization(256)
			self.bn4 = L.BatchNormalization(512)
			self.bn5 = L.BatchNormalization(512)

			self.fc6 = L.Linear(None, 4096)
			self.fc7 = L.Linear(None, 4096)
			self.fc8 = L.Linear(None, 2)

	def __call__(self, x):
		# CNN
		self.h_conv1 = self.bn1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
		self.h_pool1 = F.max_pooling_2d(self.h_conv1, (2,2), stride=2)	#64->32

		self.h_conv2 = self.bn2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.h_pool1)))))
		self.h_pool2 = F.max_pooling_2d(self.h_conv2, (2,2), stride=2)	#32->16

		self.h_conv3 = self.bn3(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(self.h_pool2)))))))
		self.h_pool3 = F.max_pooling_2d(self.h_conv3, (2,2), stride=2)	#16->8

		self.h_conv4 = self.bn4(F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(self.h_pool3)))))))
		self.h_pool4 = F.max_pooling_2d(self.h_conv4, (2,2), stride=2)	#8->4

		self.h_conv5 = self.bn5(F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(self.h_pool4)))))))
		self.h_pool5 = F.max_pooling_2d(self.h_conv5, (2,2), stride=2)	#4->2

		# fc
		self.fc_6 = F.relu(self.fc6(self.h_pool5.reshape(-1, 512*2**2)))
		self.fc_7 = F.relu(self.fc7(F.dropout(self.fc_6, ratio=.5)))
		self.fc_8 = F.relu(self.fc8(F.dropout(self.fc_7, ratio=.5)))

		return self.fc_8


class VGG_ReNorm(Chain):

	def __init__(self):
		super(VGG_ReNorm, self).__init__()
		with self.init_scope():
			# Convolution
			self.conv1_1 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)    #64->64
			self.conv1_2 = L.Convolution2D(None, 64, (3,3), stride=1, pad=1)    #64->64
			self.conv2_1 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#32->32
			self.conv2_2 = L.Convolution2D(None, 128, (3,3), stride=1, pad=1)	#32->32
			self.conv3_1 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv3_2 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv3_3 = L.Convolution2D(None, 256, (3,3), stride=1, pad=1)	#16->16
			self.conv4_1 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#8->8
			self.conv4_2 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#8->8
			self.conv4_3 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#8->8
			self.conv5_1 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv5_2 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4
			self.conv5_3 = L.Convolution2D(None, 512, (3,3), stride=1, pad=1)	#4->4

			# BatchNormalize
			self.bn1 = L.BatchRenormalization(64)
			self.bn2 = L.BatchRenormalization(128)
			self.bn3 = L.BatchRenormalization(256)
			self.bn4 = L.BatchRenormalization(512)
			self.bn5 = L.BatchRenormalization(512)

			self.fc6 = L.Linear(None, 4096)
			self.fc7 = L.Linear(None, 4096)
			self.fc8 = L.Linear(None, 2)

	def __call__(self, x):
		# CNN
		self.h_conv1 = self.bn1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
		self.h_pool1 = F.max_pooling_2d(self.h_conv1, (2,2), stride=2)	#64->32

		self.h_conv2 = self.bn2(F.relu(self.conv2_2(F.relu(self.conv2_1(self.h_pool1)))))
		self.h_pool2 = F.max_pooling_2d(self.h_conv2, (2,2), stride=2)	#32->16

		self.h_conv3 = self.bn3(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(self.h_pool2)))))))
		self.h_pool3 = F.max_pooling_2d(self.h_conv3, (2,2), stride=2)	#16->8

		self.h_conv4 = self.bn4(F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(self.h_pool3)))))))
		self.h_pool4 = F.max_pooling_2d(self.h_conv4, (2,2), stride=2)	#8->4

		self.h_conv5 = self.bn5(F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(self.h_pool4)))))))
		self.h_pool5 = F.max_pooling_2d(self.h_conv5, (2,2), stride=2)	#4->2

		# fc
		self.fc_6 = F.relu(self.fc6(self.h_pool5.reshape(-1, 512*2**2)))
		self.fc_7 = F.relu(self.fc7(F.dropout(self.fc_6, ratio=.6)))
		self.fc_8 = F.relu(self.fc8(F.dropout(self.fc_7, ratio=.6)))

		return self.fc_8


class MLP(Chain):

    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 512)
            self.l2 = L.Linear(None, 512)
            self.l3 = L.Linear(None, 512)
            self.output = L.Linear(None, 2)

    def __call__(self, x):
        self.h1 = F.relu(self.l1(x))
        self.h2 = F.relu(self.h1)
        self.h3 = F.dropout(F.relu(self.h2), ratio=.5)
        self.y = self.output(self.h3)
        return self.y
