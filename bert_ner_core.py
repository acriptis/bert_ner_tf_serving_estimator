from typing import List, Tuple, Union, Dict, Sized, Sequence, Optional
from bert.modeling import BertConfig, BertModel

import tensorflow as tf
import numpy as np
from settings import BERT_CONFIG_PATH, BERT_MODEL_PATH, N_TAGS
from settings import SEQ_LEN
n_tags = N_TAGS
LEARNING_RATE = 1e-6

def get_all_dimensions(batch: Sequence, level: int = 0, res: Optional[List[List[int]]] = None) -> List[List[int]]:
    if not level:
        res = [[len(batch)]]
    if len(batch) and isinstance(batch[0], Sized) and not isinstance(batch[0], str):
        level += 1
        if len(res) <= level:
            res.append([])
        for item in batch:
            res[level].append(len(item))
            get_all_dimensions(item, level, res)
    return res


def get_dimensions(batch) -> List[int]:
    """"""
    return list(map(max, get_all_dimensions(batch)))

def zero_pad(batch, zp_batch=None, dtype=np.float32, padding=0):
    if zp_batch is None:
        dims = get_dimensions(batch)
        zp_batch = np.ones(dims, dtype=dtype) * padding
    if zp_batch.ndim == 1:
        zp_batch[:len(batch)] = batch
    else:
        for b, zp in zip(batch, zp_batch):
            zero_pad(b, zp)
    return zp_batch


def token_from_subtoken(units: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """ Assemble token level units from subtoken level units
    Args:
        units: tf.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
        mask: mask of startings of new tokens. Example: for tokens
                [[`[CLS]` `My`, `capybara`, `[SEP]`],
                [`[CLS]` `Your`, `aar`, `##dvark`, `is`, `awesome`, `[SEP]`]]
            the mask will be
                [[0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0]]
    Returns:
        word_level_units: Units assembled from ones in the mask. For the
            example above this units will correspond to the following
                [[`My`, `capybara`],
                [`Your`, `aar`, `is`, `awesome`,]]
            the shape of this thesor will be [batch_size, TOKEN_seq_length, n_features]
    """
    shape = tf.cast(tf.shape(units), tf.int64)
    bs = shape[0]
    nf = shape[2]
    nf_int = units.get_shape().as_list()[-1]

    # numer of TOKENS in each sentence
    token_seq_lenghs = tf.cast(tf.reduce_sum(mask, 1), tf.int64)
    # for a matrix m =
    # [[1, 1, 1],
    #  [0, 1, 1],
    #  [1, 0, 0]]
    # it will be
    # [3, 2, 1]

    n_words = tf.reduce_sum(token_seq_lenghs)
    # n_words -> 6

    max_token_seq_len = tf.reduce_max(token_seq_lenghs)
    max_token_seq_len = tf.cast(max_token_seq_len, tf.int64)
    # max_token_seq_len -> 3

    idxs = tf.where(mask)
    # for the matrix mentioned above
    # tf.where(mask) ->
    # [[0, 0],
    #  [0, 1]
    #  [0, 2],
    #  [1, 1],
    #  [1, 2]
    #  [2, 0]]

    sample_id_in_batch = tf.pad(idxs[:, 0], [[1, 0]])
    # for indices
    # [[0, 0],
    #  [0, 1]
    #  [0, 2],
    #  [1, 1],
    #  [1, 2],
    #  [2, 0]]
    # it will be
    # [0, 0, 0, 0, 1, 1, 2]
    # padding is for computing change from one sample to another in the batch

    a = tf.cast(tf.not_equal(sample_id_in_batch[1:], sample_id_in_batch[:-1]), tf.int64)
    # for the example above the result of this line will be
    # [0, 0, 0, 1, 0, 1]
    # so the number of the sample in batch changes only in the last word element

    q = a * tf.cast(tf.range(n_words), tf.int64)
    # [0, 0, 0, 3, 0, 5]

    count_to_substract = tf.pad(tf.boolean_mask(q, q), [(1, 0)])
    # [0, 3, 5]

    new_word_indices = tf.cast(tf.range(n_words), tf.int64) - tf.gather(count_to_substract,
                                                                        tf.cumsum(a))
    # tf.range(n_words) -> [0, 1, 2, 3, 4, 5]
    # tf.cumsum(a) -> [0, 0, 0, 1, 1, 2]
    # tf.gather(count_to_substract, tf.cumsum(a)) -> [0, 0, 0, 3, 3, 5]
    # new_word_indices -> [0, 1, 2, 3, 4, 5] - [0, 0, 0, 3, 3, 5] = [0, 1, 2, 0, 1, 0]
    # this is new indices token dimension

    n_total_word_elements = tf.cast(bs * max_token_seq_len, tf.int32)
    x_mask = tf.reduce_sum(
        tf.one_hot(idxs[:, 0] * max_token_seq_len + new_word_indices, n_total_word_elements), 0)
    x_mask = tf.cast(x_mask, tf.bool)
    # to get absolute indices we add max_token_seq_len:
    # idxs[:, 0] * max_token_seq_len -> [0, 0, 0, 1, 1, 2] * 2 = [0, 0, 0, 3, 3, 6]
    # idxs[:, 0] * max_token_seq_len + new_word_indices ->
    # [0, 0, 0, 3, 3, 6] + [0, 1, 2, 0, 1, 0] = [0, 1, 2, 3, 4, 6]
    # total number of words in the batch (including paddings)
    # bs * max_token_seq_len -> 3 * 2 = 6
    # tf.one_hot(...) ->
    # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    #  x_mask -> [1, 1, 1, 1, 1, 0, 1, 0, 0]

    # full_range -> [0, 1, 2, 3, 4, 5, 6, 7, 8]
    full_range = tf.cast(tf.range(bs * max_token_seq_len), tf.int32)

    x_idxs = tf.boolean_mask(full_range, x_mask)
    # x_idxs -> [0, 1, 2, 3, 4, 6]

    y_mask = tf.math.logical_not(x_mask)
    y_idxs = tf.boolean_mask(full_range, y_mask)
    # y_idxs -> [5, 7, 8]

    # get a sequence of units corresponding to the start subtokens of the words
    # size: [n_words, n_features]
    els = tf.gather_nd(units, idxs)

    # prepare zeros for paddings
    # size: [batch_size * TOKEN_seq_length - n_words, n_features]
    paddings = tf.zeros(tf.stack([tf.reduce_sum(max_token_seq_len - token_seq_lenghs), nf], 0))

    tensor_flat = tf.dynamic_stitch([x_idxs, y_idxs], [els, paddings])
    # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]

    tensor = tf.reshape(tensor_flat, tf.stack([bs, max_token_seq_len, nf_int], 0))
    # tensor_flat -> [[x, x, x],
    #                 [x, x, 0],
    #                 [x, 0, 0]]

    return tensor

def bert_ner_core(input_ids_ph, input_masks_ph, y_masks_ph, y_ph, train_mode=True):
    ############ GRAPH ####################################################################
    seq_lengths = tf.reduce_sum(y_masks_ph, axis=1)

    bert = BertModel(config=BertConfig.from_json_file(BERT_CONFIG_PATH),
                     is_training=False,
                     input_ids=input_ids_ph,
                     input_mask=input_masks_ph,
                     token_type_ids=tf.zeros_like(input_ids_ph, dtype=tf.int32),
                     use_one_hot_embeddings=False)

    encoder_layers = [bert.all_encoder_layers[-1]]
    encoder_layers_count = len(encoder_layers)

    with tf.variable_scope('ner'):
        layer_weights = tf.unstack(tf.get_variable('layer_weights_',
                                                   shape=encoder_layers_count,
                                                   initializer=tf.ones_initializer(),
                                                   trainable=False) / encoder_layers_count
                                   )

        units = sum(w * l for w, l in zip(layer_weights, encoder_layers))

        # TODO: maybe add one more layer?
        logits_raw = tf.layers.dense(units, units=n_tags, name="output_dense")

        logits = token_from_subtoken(logits_raw, y_masks_ph)
        max_seq_length = tf.math.reduce_max(seq_lengths)

        # TODO do we need it?
        one_hot_max_len = tf.one_hot(seq_lengths - 1, max_seq_length)
        # one_hot_max_len = tf.one_hot(seq_lengths - 1, max_seq_length, dtype=tf.int64)
        tag_mask = tf.cumsum(one_hot_max_len[:, ::-1], axis=1)[:, ::-1]

        # max_seq_length = tf.constant(465)
        transition_params = tf.get_variable('Transition_Params',
                                            shape=[n_tags, n_tags],
                                            initializer=tf.zeros_initializer())

        logits_shape = tf.cast(tf.shape(logits), tf.int32)
        logits_size = logits_shape[1]

        global_max_seq_len = tf.constant(SEQ_LEN)

        left_pad_size = global_max_seq_len - logits_size

        logits_padded = tf.pad(logits, [[0, 0], [0, left_pad_size], [0, 0]], "CONSTANT")
        y_predictions = tf.argmax(logits, -1)
        y_probas = tf.nn.softmax(logits, axis=2)

        if train_mode:

            log_likelihood, transition_params = \
                tf.contrib.crf.crf_log_likelihood(logits_padded,
                                                  y_ph,
                                                  # y_ph_stripped,
                                                  seq_lengths,
                                                  transition_params)

            loss_tensor = -log_likelihood

    with tf.variable_scope("loss"):
        if train_mode:
            y_mask = tf.cast(tag_mask, tf.float32)
            loss_op = tf.reduce_mean(loss_tensor)

        # #####################################################
        # TODO where it from?:
        # padding_mask = tf.sequence_mask(seq_lengths, max_seq_length)
        # pred, _ = tf.contrib.crf.crf_decode(tranformed_logits, transition_params, seq_lengths)
        # tags = tf.where(padding_mask, x=pred, y=tf.fill(tf.shape(pred), -1))
        # #####################################################
        # from deep pavlov
        # tags = tf.argmax(tranformed_logits, -1)
        # y_probas = tf.nn.softmax(tranformed_logits, axis=2)

    # sess.run(tf.global_variables_initializer())

    variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # saver = tf.train.Saver()
    # saver.restore(sess, '../resources/ner_rus_bert/model')
    # print("tf.GraphKeys.TRAINABLE_VARIABLES")
    # print(variables_to_train)
    if train_mode:
        # opt_scope = tf.variable_scope('Optimizer', reuse=tf.AUTO_REUSE)
        with tf.variable_scope('Optimizer'):
            # For batch norm it is necessary to update running averages
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
                optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

                # print(variables_to_train)
                grads_and_vars = optimizer.compute_gradients(loss_op, var_list=variables_to_train)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        return y_predictions, train_op, loss_op
    else:
        return y_predictions, None, None
