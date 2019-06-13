import tensorflow as tf
# from tf_metrics import f1

from settings import BERT_MODEL_PATH, TAGS_MAP
from bert_ner_core import bert_ner_core


def model_fn(features, labels, mode, params):
    input_ids_ph = tf.cast(features['input_ids'], tf.int32)
    input_masks_ph = tf.cast(features['input_masks'], tf.int32)
    y_masks_ph = tf.cast(features['y_masks'], tf.int32)
    y_ph = labels

    # ##############################################################################################

    if mode == tf.estimator.ModeKeys.PREDICT:
        y_predictions, train_op, loss_op = bert_ner_core(input_ids_ph, input_masks_ph, y_masks_ph,
                                                         y_ph, train_mode=False)
        predictions_dict = {
            'tags': y_predictions,
            'tags_str': tf.gather(list(TAGS_MAP.keys()), y_predictions),
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            # specifies format of output in TF Serving:
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions_dict['tags_str'])
            }
        )

    # Train or Eval case goes here:
    y_predictions, train_op, loss_op = bert_ner_core(input_ids_ph, input_masks_ph, y_masks_ph,
                                                     y_ph, train_mode=True)
    labels_shape = tf.cast(tf.shape(labels), tf.int32)
    labels_size = labels_shape[1]
    ys_shape = tf.cast(tf.shape(y_predictions), tf.int32)
    ys_size = ys_shape[1]
    left_pad = labels_size - ys_size
    y_predictions_padded = tf.pad(y_predictions, [[0, 0], [0, left_pad]], "CONSTANT")

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=y_predictions_padded,
                                   name='acc_op')
    precision = tf.metrics.precision(labels=labels,
                                     predictions=y_predictions_padded,
                                     name='precision_op')
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': tf.metrics.recall(labels=labels,
                                    predictions=y_predictions_padded,
                                    name='recall_op'),
        # TODO why f1 doesnt work with error:
        # tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed: [predictions must be in [0, 1]] [Condition x <= y did not hold element-wise:x (Pad:0) = ] [[0 0 0...]...] [y (f1_op/Cast_1:0) = ] [1]
        # 	 [[node f1_op/assert_less_equal/Assert/AssertGuard/Assert (defined at /home/alx/Workspace/dp_bert_ner/get_bert_estimator.py:59) ]]

        # 'f1': tf.contrib.metrics.f1_score(labels=labels,
        #                                   predictions=y_predictions_padded,
        #                                   name='f1_op',
        #                                   num_thresholds=1,
        #                                   ),
        # 'f1': f1(labels=labels,
        #          predictions=y_predictions_padded,
        #          num_classes=len(TAGS_MAP.keys())),
    }
    for metric_name, op in metrics.items():
        tf.summary.scalar(metric_name, op[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss_op, eval_metric_ops=metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=train_op,
            loss=loss_op
        )

    raise Exception('Unsupported Mode: %s!' % mode)


my_checkpointing_config = tf.estimator.RunConfig(
    # save_checkpoints_secs=60,
    save_checkpoints_steps=1,
    keep_checkpoint_max=20,
    save_summary_steps=5,
)


def get_estimator():
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params={},
        model_dir=BERT_MODEL_PATH,
        config=my_checkpointing_config
    )
