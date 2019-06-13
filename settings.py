import os
RESOURCES_PATH = "./res/BERT_body"

BERT_MODEL_PATH = os.path.join(RESOURCES_PATH, 'ner_rus_bert/model')
BERT_CONFIG_PATH = os.path.join(RESOURCES_PATH, 'rubert_cased_L-12_H-768_A-12_v1/bert_config.json')
VOCAB_PATH = os.path.join(RESOURCES_PATH, 'rubert_cased_L-12_H-768_A-12_v1/vocab.txt')

# max length of sentence:
SEQ_LEN = 465

TAGS_MAP = {
    'O': 0,
    'I-PER': 1,
    'B-PER': 2,
    'I-ORG': 3,
    'B-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6
}

# number of tag types
N_TAGS = len(TAGS_MAP.keys())

# marker for bio2tf.py for labeling places in sample that are out of scope of current sentence
# (right part of vector of labels from position of last token up to SEQ_LEN)
# OUT_OF_SENTENCE_CODE = -1
OUT_OF_SENTENCE_CODE = 0

BATCH_SIZE = 19