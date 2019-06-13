#!/usr/bin/env python3

import tensorflow as tf
# from tensorflow.contrib.learn import ModeKeys
# import ner as n

# from ner import _bert_encode, _bert_decode, _crf

from input_fn import input_fn
from get_bert_estimator import get_estimator

if __name__ == "__main__":
    # input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_indices_ph')
    # input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_mask_ph')
    # y_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_mask_ph')

    e = get_estimator()
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    # Unfortunately we can not restore variables here:
    # saver = tf.train.Saver()
    # saver = tf.train.Saver(
    #     var_list=[v for v in tf.trainable_variables() if not v.name.startswith('Optimizer')])
    # saver.restore(sess, '../resources/ner_rus_bert/model')

    import time

    start = time.clock()
    print("start at:")
    print(start)
    # your code here
    import functools
    predict_inpf = functools.partial(input_fn, tfrecord_ds_path='data/micro_test.tfrecord')

    for w in e.predict(predict_inpf):
        print(w)
    fintime = time.clock()
    tdelta = fintime - start
    print("start at")
    print(start)

    print("fintime")
    print(fintime)
    print("Total timedelta:")
    print(time.clock() - start)
