#!/usr/bin/env python3
"""
Usage:
    Pipe BIO markup datafile through bio2tf.py script:
        `cat data/train.txt | ./bio2tf.py data/train.tfrecord`

    Provide path to target tfrecord datafile as argument.
"""
import sys
import argparse
import tensorflow as tf
from tensorflow.train import BytesList, Feature, Features, Example
from bert_ner_preprocessor import tokenize

parser = argparse.ArgumentParser(description='Convert BIO- file to tfrecords file')
parser.add_argument('output', type=str, metavar='OUTPUT',
                    help='tf record file witch will be written')
out_file = parser.parse_args().output

document_words = []
document_tags = []
documents_counter = 0
sentences_counter = 0

with tf.python_io.TFRecordWriter(out_file) as writer:
    def rollout_to_protobuf(document_words, document_tags):
        print("document_words")
        print(document_words)
        print("document_tags")
        print(document_tags)
        input_ids, input_masks, y_masks, text, ys = tokenize([document_words], [document_tags])

        print("text")
        print(text)
        example_proto = Example(features=Features(feature={
            'text': Feature(
                bytes_list=tf.train.BytesList(value=[w.encode('utf-8') for w in text[0]])),
            'input_ids': Feature(int64_list=tf.train.Int64List(value=input_ids)),
            'input_masks': Feature(int64_list=tf.train.Int64List(value=input_masks)),
            'y_masks': Feature(int64_list=tf.train.Int64List(value=y_masks)),
            'labels': Feature(int64_list=tf.train.Int64List(value=ys))
        }))
        writer.write(example_proto.SerializeToString())

    for line in sys.stdin:

        print(f"************")
        print(f"{line}")
        if not line.strip():
            sentences_counter+=1

            print("NOLINE")
            if document_words:
                # accumulate
                rollout_to_protobuf(document_words, document_tags)
                document_words = []
                document_tags = []
            continue
        if '<DOCSTART>' in line:
            documents_counter+=1
        if ('<DOCSTART>' in line and document_words and document_tags):
            # if ('<DOCSTART>' in line and document_words and document_tags) or '.' in line:
            # if '<DOCSTART>' in line and document_words and document_tags:
            print("document_words:")
            print(document_words)

            rollout_to_protobuf(document_words, document_tags)
            document_words = []
            document_tags = []
        elif len(line.split('\t')) == 2:
            line = line.strip()
            word, label = line.split('\t')
            document_words.append(word)
            document_tags.append(label)

print("Number of documents")
print(documents_counter)
print("Number of sentences")
print(sentences_counter)
