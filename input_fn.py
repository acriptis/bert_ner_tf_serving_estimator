import tensorflow as tf
DATASET_FILENAME = 'data/micro_test.tfrecord'
from settings import BATCH_SIZE


def parse(serialized_example):
    """
    Function adapts features for the Estimator format
    :param serialized_example:
    :return:
    """
    features_spec = {
        'input_ids': tf.VarLenFeature(tf.int64),
        'input_masks': tf.VarLenFeature(tf.int64),
        'y_masks': tf.VarLenFeature(tf.int64),
        'labels': tf.VarLenFeature(tf.int64),
        'text': tf.VarLenFeature(tf.string),
    }

    example = tf.parse_single_example(serialized_example, features_spec)

    return ({
                'input_ids': tf.sparse.to_dense(example['input_ids']),
                'input_masks': tf.sparse.to_dense(example['input_masks']),
                'y_masks': tf.sparse.to_dense(example['y_masks']),
                'text': example['text'],
            },
            tf.sparse.to_dense(example['labels']))


def input_fn(tfrecord_ds_path=None, batch_size=None):
    """
    Input function accepts zero argument case for compatibility with tensorflow input_fn
    https://www.tensorflow.org/guide/datasets_for_estimators#try_it_out:
        Estimators expect an input_fn to take no arguments
    :param tfrecord_ds_path: path to dataset in tfrecord format
    :return: TFRecordDataset
    """
    if not tfrecord_ds_path:
        tfrecord_ds_path = DATASET_FILENAME
    if not batch_size:
        batch_size = BATCH_SIZE

    dataset = tf.data.TFRecordDataset([tfrecord_ds_path]).map(
        lambda record: parse(record)).shuffle(100).batch(batch_size)
        # lambda record: parse(record)).batch(batch_size)

        # lambda record: parse(record)).shuffle(100).batch(12)
        # 100).repeat().batch(12)

    return dataset


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders
    https://www.tensorflow.org/guide/saved_model#prepare_serving_inputs
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    # TODO adapt me
    input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_indices_ph')
    input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_mask_ph')
    y_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_mask_ph')
    receiver_tensors = {
        'input_ids': input_ids_ph,
        'input_masks': input_masks_ph,
        'y_masks': y_masks_ph,
    }

    return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)
