# -*- coding: utf-8 -*-
import threading
from ops import *
import os
from glob import glob
import numpy as np
import tensorflow as tf
import cv2
from sklearn.utils import shuffle
from random import randint


def download(dataset_name):
    datasets_dir = './datasets/'
    mkdir(datasets_dir)
    URL='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz' % (dataset_name)
    TAR_FILE='./datasets/%s.tar.gz' % (dataset_name)
    TARGET_DIR='./datasets/%s/' % (dataset_name)
    os.system('wget -N %s -O %s' % (URL, TAR_FILE))
    os.mkdir(TARGET_DIR)
    os.system('tar -zxf %s -C ./datasets/' % (TAR_FILE))
    os.remove(TAR_FILE)


class DLoader(object):
    def __init__(self, config, resize=True):
        self.batch_size = config['batch_size']
        self.thread_num = config['thread_num']
        print("Batch size: %d, Thread num: %d" % (self.batch_size, self.thread_num))

        if not os.path.isdir(config['data_root']):
            print('bad data root. update config')
        
        self.resize = resize
        self.img_inp_shape = np.array(config['img_inp_shape'])
        self.min_size = config['min_size']
        self.img_pad_val = config['img_pad_val']
        self.label_pad_val = config['label_pad_val']
        
        self.data = shuffle(glob(config['data_root'] + '*'))
        self.train_data = self.data[:int(0.8*len(self.data))]
        self.val_data = self.data[int(0.8*len(self.data)):int(0.9*len(self.data))]
        self.test_data = self.data[int(0.9*len(self.data)):]
        self.data_size = len(self.train_data)
        self.data_indice = range(self.data_size - 1)
        

        
        print("load dataset done")
        print('data size: %d' % self.data_size)
        self.img_shape = list(self.img_inp_shape)
        self.label_shape = list(self.img_inp_shape)#[:2] + [2]
        print('in shape:', self.img_shape, 'label shape:', self.label_shape)
        self.fine_size = self.img_shape[0]
        self.load_size = self.fine_size + int(0.11*self.fine_size)

        self.img_data = tf.placeholder(tf.float32, shape=[None] + self.img_shape)
        self.label_data = tf.placeholder(tf.float32, shape=[None] + self.label_shape)
        self.queue = tf.FIFOQueue(shapes=[self.label_shape, self.img_shape],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=2000)
        self.enqueue_ops = self.queue.enqueue_many([self.label_data, self.img_data])
    
    def load_img_label(self, folder):
        img_path = os.path.join(folder, os.path.split(folder)[1]+ '.jpg')
        label_path = os.path.join(folder, os.path.split(folder)[1]+ '_mask.png')
        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        return img, label
        
    def resize_img_label(self, img, label):
        org_size = img.shape
        if self.resize: 
            scale = np.min(self.img_inp_shape[:2]/np.array(org_size)[:2])
            img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            pad = self.img_inp_shape[:2] - img.shape[:2]
        else:
            scale = 1
            if np.max(org_size[:2]) > self.img_inp_shape[0]:
                scale = self.img_inp_shape[0]/np.max(org_size[:2])
            elif np.min(org_size[:2]) < self.min_size:
                scale = self.min_size/np.min(org_size[:2])
            if scale != 1:
                img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            pad = (64 - np.array(img.shape)[:2] % 64) % 64
        return img, label, pad
    
    def pad_img_label(self, img, label, pad):
        img = np.pad(img, ((0,pad[0]), (0,pad[1]), (0,0)), 'constant', constant_values=self.img_pad_val)
        label = np.pad(label, ((0,pad[0]), (0,pad[1]), (0,0)), 'constant', constant_values=self.label_pad_val)
        return img, label
        
    def get_img_label(self, folder, augment=False):
        img, label = self.load_img_label(folder)
        img, label, pad = self.resize_img_label(img, label)
        img, label = self.pad_img_label(img, label, pad)
        img = img_shift(img)
        label = img_shift(label)
        if augment:
            img, label = self.img_augment(img, label)
        return img, label
    
    def img_augment(self, img, label):
        img = cv2.resize(img, (self.load_size, self.load_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.load_size, self.load_size), interpolation=cv2.INTER_NEAREST)

        h1 = int(np.ceil(np.random.uniform(0, self.load_size-self.fine_size)))
        w1 = int(np.ceil(np.random.uniform(0, self.load_size-self.fine_size)))
        img = img[h1:h1+self.fine_size, w1:w1+self.fine_size]
        label = label[h1:h1+self.fine_size, w1:w1+self.fine_size]
        if np.random.random() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
        return img, label
    
    def batch_iterator(self, augment=True, debug=False, shuff=True):
        samp_list = self.train_data
        while True:
            if shuff:
                samp_list = shuffle(samp_list)
            for i in range(len(self.data_indice)//self.batch_size):
                if debug:
                    print(i, 'of', len(self.data_indice)//self.batch_size)
                img_batch, label_batch = [], []
                for j in range(i*self.batch_size, (i+1)*self.batch_size):
                    img, label = self.get_img_label(samp_list[j], augment=augment)
                    label_batch.append(label)
                    img_batch.append(img)
                yield np.array(label_batch), np.array(img_batch)
    
    def get_inputs(self):
        labels, imgs = self.queue.dequeue_many(self.batch_size)
        return labels, imgs
    
    def thread_main(self, sess):
        for labels, imgs in self.batch_iterator():
            sess.run(self.enqueue_ops, feed_dict={self.label_data: imgs , self.img_data: labels})
            sess.run(self.enqueue_ops, feed_dict={self.label_data: imgs , self.img_data: labels})
    
    def start_threads(self, sess):
        threads = []
        for n in range(self.thread_num):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads
    
    def get_size(self):
        return self.data_size

    def get_shape(self):
        return self.img_shape, self.label_shape
    
    def get_batch_imgs(self, train=False, ret_names=False):
        test_imgs, test_labels, folders = [], [], []
        for i in range(self.batch_size):
            if train:
                folder = self.train_data[randint(0, len(self.train_data)-1)]
                augment=True
            else:
                folder = self.val_data[randint(0, len(self.val_data)-1)]
                augment=False
            folders.append(folder)
            test_img, test_label = self.get_img_label(folder, augment=augment)
            test_imgs.append(test_img)
            test_labels.append(test_label)
        if ret_names:
            return test_imgs, test_labels, folders
        else:
            return np.array(test_imgs), np.array(test_labels)
    
    
    
    
class Dataset(object):
    def __init__(self, dataset, is_test=False, batch_size=4, crop_width=256, thread_num=1):
        self.batch_size = batch_size
        self.thread_num = thread_num
        print("Batch size: %d, Thread num: %d" % (batch_size, thread_num))
        datasetDir = './datasets/{}'.format(dataset)
        if not os.path.isdir(datasetDir):
            download(dataset)
        dataDir = datasetDir + '/train'
        data = glob((dataDir + '/*.jpg').format(dataset))
        self.data_size = min(400, len(data))
        self.data_indice = range(self.data_size - 1)
        self.dataDir = dataDir
        self.is_test = is_test
        self.dataset = []
        for i in range(1, self.data_size):
            img, label = load_image(self.dataDir + '/%d.jpg' % i)

            self.dataset.append((img, label))
        print("load dataset done")
        print('data size: %d' % len(self.dataset))
        self.img_shape = list(self.dataset[0][0].shape)
        self.label_shape = list(self.dataset[0][1].shape)
        self.fine_size = self.img_shape[0]
        self.crop_width = self.fine_size
        self.load_size = self.fine_size + 30

        self.img_data = tf.placeholder(tf.float32, shape=[None] + self.img_shape)
        self.label_data = tf.placeholder(tf.float32, shape=[None] + self.label_shape)
        self.queue = tf.FIFOQueue(shapes=[self.label_shape, self.img_shape],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=2000)
        self.enqueue_ops = self.queue.enqueue_many([self.label_data, self.img_data])

        
    def batch_iterator(self):
        while True:
            shuffle_indices = np.random.permutation(self.data_indice)
            for i in range(len(self.data_indice) // self.batch_size):
                img_batch = []
                label_batch = []
                for j in range(i*self.batch_size, (i+1)*self.batch_size):
                    [img,label] = self.dataset[shuffle_indices[j]]
                    #img = self.dataset[shuffle_indices[j]][0]
                    img, label = img_preprocess(img, label, self.fine_size, self.load_size)
                    label_batch.append(label)
                    img_batch.append(img)
                yield np.array(label_batch), np.array(img_batch)
    
    def get_inputs(self):
        labels, imgs = self.queue.dequeue_many(self.batch_size)
        return labels, imgs
    
    def thread_main(self, sess):
        for labels, imgs in self.batch_iterator():
            sess.run(self.enqueue_ops, feed_dict={self.label_data: labels , self.img_data: imgs})
            sess.run(self.enqueue_ops, feed_dict={self.label_data: labels , self.img_data: imgs})
    
    def start_threads(self, sess):
        threads = []
        for n in range(self.thread_num):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads
    
    def get_size(self):
        return self.data_size

    def get_shape(self):
        return self.img_shape, self.label_shape
    
