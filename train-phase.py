# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

import tensorflow as tf
#from model import Model
from model import Model
import os
import shutil
from data import Data
from preprocess import to_spectrogram, get_magnitude, get_phase
from utils import Diff
from config import TrainConfig
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

os.environ['CUDA_VISIBLE_DEVICES']='6'

# TODO multi-gpu
def train():
    # Model
    model = Model()

    # Loss, Optimizer
    ## Variable to keep track of how many times the graph has been run
    num_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='num_epoch')
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=TrainConfig.LR).minimize(loss_fn, global_step=global_step)

    # Summaries, possible to visualize data's distribution in TensorBoard
    #summary_op = summaries(model, loss_fn)

    # Input source
    print('..........Load Data............')
    data = Data(TrainConfig.DATA_PATH)
    data.load_data()
    total_batch = data.total_batches()
    print('len of data:{}'.format(len(data.wavfiles)))

    with tf.Session(config=TrainConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load_state(sess, TrainConfig.CKPT_PATH)

        #writer = tf.summary.FileWriter(TrainConfig.GRAPH_PATH, sess.graph)

        loss = Diff()
        step = 0
        epoch_loss = []
        for epoch in range(num_epoch.eval(), TrainConfig.FINAL_EPOCH): # changed xrange to range for py3


            intermediate_loss = []

            for i in range(0, total_batch):
                mixed_wav, src1_wav, src2_wav, _ = data.next_batch(TrainConfig.SECONDS)   #get batch

                mixed_spec = to_spectrogram(mixed_wav)   #stft
                mixed_mag = get_magnitude(mixed_spec)    #abs, mag
                mixed_phase = get_phase(mixed_spec)

                src1_spec, src2_spec = to_spectrogram(src1_wav), to_spectrogram(src2_wav)
                src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)

                src1_batch, _ = model.spec_to_batch(src1_mag)
                src2_batch, _ = model.spec_to_batch(src2_mag)
                mixed_mag_batch, _ = model.spec_to_batch(mixed_mag)
                mixed_phase_batch, _ = model.spec_to_batch(mixed_phase)

                l, _ = sess.run([loss_fn, optimizer],
                                     feed_dict={model.x_mixed_mag: mixed_mag_batch, model.x_mixed_phase: mixed_phase_batch,
                                                model.y_src1: src1_batch, model.y_src2: src2_batch})
                loss.update(l)
                intermediate_loss.append(l)
                step += 1
                #print('step-{}\tepoch-{}\tbatch-{}\td_loss={:2.2f}\tloss={}'.format(step, epoch, i, loss.diff * 100, loss.value))
                #writer.add_summary(summary, global_step = step)
                
            
            print('epoch-{}\tloss={}'.format(epoch, np.mean(intermediate_loss)))
            epoch_loss.append(np.mean(intermediate_loss))
            # Save state
            if epoch % TrainConfig.CKPT_EPOCH == 0:
                tf.train.Saver().save(sess, TrainConfig.CKPT_PATH + '/checkpoint', global_step = step)
        print('epoch_loss:{}'.format(epoch_loss))  
        draw(epoch_loss)  
        #writer.close()
    

'''
def summaries(model, loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
        tf.summary.histogram('grad/' + v.name, tf.gradients(loss, v))
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('y_src1', model.y_src1)
    tf.summary.histogram('y_src2', model.y_src1)
    return tf.summary.merge_all()
'''

def setup_path():
    if TrainConfig.RE_TRAIN:
        if os.path.exists(TrainConfig.CKPT_PATH):
            shutil.rmtree(TrainConfig.CKPT_PATH)
        if os.path.exists(TrainConfig.GRAPH_PATH):
            shutil.rmtree(TrainConfig.GRAPH_PATH)
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.makedirs(TrainConfig.CKPT_PATH)

def draw(epoch_loss):
    plt.plot(range(TrainConfig.FINAL_EPOCH), epoch_loss, linewidth=2.0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.savefig("training loss.png")

# when run separately, __name__ == '__main__'
if __name__ == '__main__':
    setup_path()
    train()
