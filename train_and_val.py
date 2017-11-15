# -*- coding: utf-8 -*-
import os
import os.path

import numpy as np
import tensorflow as tf

import input_data
import VGG
import tools

#%%
IMG_W = 224
IMG_H = 224
N_CLASSES = 80
#RATIO = 0.2 # take 20% of dataset as validation data 
BATCH_SIZE = 32
learning_rate = 0.001
CAPACITY = 2000
MAX_STEP = 15000   # it took me about one hour to complete the training.
IS_PRETRAIN = True


#%%   Training
def train():
    
    pre_trained_weights = './vgg16_pretrain/vgg16.npy'
    train_data_dir = './data/train/scene_train_images_20170904/'
    train_label_json = './data/train/scene_train_annotations_20170904.json'
    val_data_dir = './data/val/scene_validation_images_20170908/'
    val_label_json = './data/val/scene_validation_annotations_20170908.json'
    train_log_dir = './logs/train/'
#    val_log_dir = './logs/val/'
    
    with tf.name_scope('input'):
        
        tra_images, tra_labels = input_data.get_files(train_label_json, train_data_dir)
        
        tra_image_batch, tra_label_batch =  input_data.get_batch(tra_images, 
                                                                 tra_labels,
                                                                 IMG_W,
                                                                 IMG_H,
                                                                 BATCH_SIZE,
                                                                 CAPACITY,
                                                                 N_CLASSES)
        
        val_images, val_labels = input_data.get_files(val_label_json, val_data_dir)
        val_image_batch, val_label_batch = input_data.get_batch(val_images,
                                                                val_labels,
                                                                IMG_W,
                                                                IMG_H,
                                                                BATCH_SIZE,
                                                                CAPACITY,
                                                                N_CLASSES)
        
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES]) 
    keep_prob = tf.placeholder(tf.float32)
    
    logits = VGG.VGG16N(x, N_CLASSES, keep_prob, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = tools.optimize(loss, learning_rate, my_global_step)
      
    saver = tf.train.Saver(tf.global_variables())
#    summary_op = tf.summary.merge_all()   
      
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    #load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])   


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
#    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
#    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
   
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
               
            train_images,train_labels = sess.run([tra_image_batch, tra_label_batch])
            #print(str(train_images.get_shape()))
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                           feed_dict={x: train_images, y_: train_labels, keep_prob: 0.2})            
            if step % 50 == 0 or (step + 1) == MAX_STEP:                 
#                _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
#                                                feed_dict={x: train_images, y_: train_labels}) 
                print ('Step: %d, loss: %.3f, accuracy: %.3f%%' % (step, tra_loss, tra_acc))
#                summary_str = sess.run(summary_op)
#                tra_summary_writer.add_summary(summary_str, step)
               
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                validation_images, validation_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                            feed_dict={x: validation_images, y_: validation_labels, keep_prob: 1})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))

#                summary_str = sess.run(summary_op)
#                val_summary_writer.add_summary(summary_str, step)
                   
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
               
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
       
    coord.join(threads)
    sess.close()

#%%%
if __name__ == "__main__":
    train()