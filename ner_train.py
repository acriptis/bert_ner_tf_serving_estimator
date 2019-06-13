from get_bert_estimator import get_estimator
from input_fn import input_fn
import argparse
import time
import datetime as dt
import functools
from input_fn import serving_input_receiver_fn


def main(train_dataset_path, model_save_path, training_steps, batch_size, *args, **kwargs):
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess, '../resources/ner_rus_bert/model')

    # load bert as Estimator:
    bert_ner_estimator = get_estimator()

    start = time.clock()
    start_dt = dt.datetime.now()
    print("start at:")
    print(start)
    print(start_dt)
    print("Start train")

    train_inpf = functools.partial(input_fn, tfrecord_ds_path=train_dataset_path, batch_size=batch_size)
    result = bert_ner_estimator.train(train_inpf, steps=training_steps)

    print("Fin train")
    print(result)

    fintime = time.clock()
    fin_dt = dt.datetime.now()
    print("start at")
    print(start)

    print("fintime")
    print(fintime)
    print(fin_dt)
    print("Total timedelta:")
    print(fintime - start)
    print(fin_dt - start_dt)

    # Export it (How it saves without this string?)
    print("Exporting model to %s" % model_save_path)

    # bert_ner_estimator.export_saved_model('resources/BERT_NER_ESTIMATOR_saved_model',
    bert_ner_estimator.export_saved_model(model_save_path,
                                          serving_input_receiver_fn)
    print("Fin")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', help='Path to TFRecord dataset for training', default='data/train.tfrecord', type=str)
    parser.add_argument('--model_save_path', help='Path to save the resulting model', default='res/BERT_NER_ESTIMATOR', type=str)
    parser.add_argument('--training_steps', help='Number of steps in training', default=10, type=int)
    parser.add_argument('--batch_size', help='Size of Batch', default=19, type=int)
    args = parser.parse_args()
    main(args.train_dataset, args.model_save_path, args.training_steps, args.batch_size)
