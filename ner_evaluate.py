import argparse
import functools
import tensorflow as tf
# from tensorflow.python import debug as tf_debug
from get_bert_estimator import get_estimator
from input_fn import input_fn

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to TFRecord dataset for evaluation',
                        default='data/test.tfrecord', type=str)
    # parser.add_argument('--model_save_path', help='Path to save the resulting model',
    #                     default='res/BERT_NER_ESTIMATOR', type=str)
    # parser.add_argument('--training_steps', help='Number of steps in training', default=10,
    #                     type=int)
    # args = parser.parse_args()
    args = parser.parse_args(argv[1:])
    bert_ner_estimator = get_estimator()

    # train_inpf = functools.partial(input_fn, tfrecord_ds_path='data/train.tfrecord')

    # Train the Model.
    # bert_ner_estimator.train(
    #     train_inpf,
    #     steps=args.train_steps)
    # First, let your BUILD target depend on "//tensorflow/python/debug:debug_py"
    # (You don't need to worry about the BUILD dependency if you are using a pip
    #  install of open-source TensorFlow.)
    # Create a LocalCLIDebugHook and use it as a monitor when calling fit().
    # hooks = [tf_debug.LocalCLIDebugHook()]

    print('\nStarting evaluation on dataset: %s...\n' % args.dataset)
    # Evaluate the model.
    # eval_inpf = functools.partial(input_fn, tfrecord_ds_path='data/valid.tfrecord')

    # eval_inpf = functools.partial(input_fn, tfrecord_ds_path='data/micro_test.tfrecord')
    eval_inpf = functools.partial(input_fn, tfrecord_ds_path=args.dataset)
    eval_result = bert_ner_estimator.evaluate(
        input_fn=eval_inpf,
        # hooks=hooks
    )

    print('\nAccuracy: {accuracy:0.3f}\n'.format(**eval_result))
    #
    # # Generate predictions from the model
    # expected = ['Setosa', 'Versicolor', 'Virginica']
    # predict_x = {
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # }
    #
    # predictions = classifier.predict(
    #     input_fn=lambda:iris_data.eval_input_fn(predict_x,
    #                                             labels=None,
    #                                             batch_size=args.batch_size))
    #
    # for pred_dict, expec in zip(predictions, expected):
    #     template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    #
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][class_id]
    #
    #     print(template.format(iris_data.SPECIES[class_id],
    #                           100 * probability, expec))


if __name__ == '__main__':

    # main(args.train_dataset, args.model_save_path, args.training_steps)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
