import numpy as np

mnist = './MNIST/'
m_trainim = 'train_images.npy'
m_labelim = 'train_labels.npy'
m_testim = 'test_images.npy'
m_labelim = 'test_labels.npy'

testIm = np.load(mnist + m_testim)

print testIm

