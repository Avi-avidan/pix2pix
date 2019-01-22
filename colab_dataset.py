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


class DLoader(object):
    def __init__(self, config, resize=False, augment=False, debug=False):
        self.resize = resize
        self.augment = augment
        self.debug = debug
        self.batch_size = config['batch_size']
        self.thread_num = config['thread_num']
        self.pad_divisable = config['pad_divisable']
        self.img_inp_shape = np.array(config['img_inp_shape'])
        self.min_size = config['min_size']
        self.img_pad_val = config['img_pad_val']
        self.label_pad_val = config['label_pad_val']
        
        self.pickled_data = config['train_val_lists']
        self.train_data, self.val_data, self.test_data = self.load_data()
        self.data_size = len(self.train_data)
        self.data_indice = range(self.data_size - 1)

        self.img_shape = config['img_inp_shape']
        self.label_shape = config['img_inp_shape']

        self.fine_size = config['min_size']
        self.load_size = self.fine_size + int(0.11*self.fine_size)

        self.img_data = tf.placeholder(tf.float32, shape=[None] + self.img_shape)
        self.label_data = tf.placeholder(tf.float32, shape=[None] + self.label_shape)
        self.queue = tf.FIFOQueue(shapes=[self.label_shape, self.img_shape],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=2000)
        self.enqueue_ops = self.queue.enqueue_many([self.label_data, self.img_data])
        self.print_load_done()
        
    def print_load_done(self):
        if not os.path.exists(self.pickled_data):
            print('bad data root. update config')
        else:
            print("Batch size: %d, Thread num: %d" % (self.batch_size, self.thread_num))
            print('in shape:', self.img_shape, 'label shape:', self.out_shape)
            print("load dataset done")
            print('data size: %d' % self.data_size)

    def load_data(self):
        data = pickle.load(open(self.pickled_data, 'rb'))
        return data['x_train'], data['x_val'], data['x_test']
    
    def load_img(self, img_path):
        img = imageio.imread(img_path)
        return np.expand_dims(img, axis=-1)
        
    def load_img_label(self, img_name):
        img_path = os.path.join(self.data_root, 'jpg', img_name+ '.jpg')
        label_path = os.path.join(self.data_root, 'bmp', img_name+ '.bmp')
        img = self.load_img(img_path)
        label = self.load_img(label_path)
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
            pad = -np.array(img.shape)[:2] % self.pad_divisable
        return img, label, pad
    
    def pad_img_label(self, img, label, pad):
        img = np.pad(img, ((0,pad[0]), (0,pad[1]), (0,0)), 'constant', constant_values=self.img_pad_val)
        label = np.pad(label, ((0,pad[0]), (0,pad[1]), (0,0)), 'constant', constant_values=self.label_pad_val)
        return img, label
        
    def get_img_label(self, folder, augment=False):
        img, label = self.load_img_label(folder)
        img, label, pad = self.resize_img_label(img, label)
        if np.any(pad == 0):
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
        if np.random.random() > 0.5:
            img = np.flipud(img)
            label = np.flipud(label)
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
        if self.batch_size > 1:
            labels, imgs = self.queue.dequeue_many(self.batch_size)
        else:
            labels, imgs = self.queue.dequeue()
            labels = tf.expand_dims(labels, 0)
            imgs = tf.expand_dims(imgs, 0)
        return imgs, labels
    
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