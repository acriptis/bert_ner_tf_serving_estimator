from typing import List, Tuple, Union, Dict, Sized, Sequence, Optional
from bert.tokenization import FullTokenizer
import re
from logging import getLogger
import numpy as np

from settings import VOCAB_PATH, SEQ_LEN, TAGS_MAP, OUT_OF_SENTENCE_CODE


class BertNerPreprocessor:
    """Takes tokens and splits them into bert subtokens, encode subtokens with their indices.
    Creates mask of subtokens (one for first subtoken, zero for later subtokens).
    If tags are provided, calculate tags for subtokens.
    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: replace token to <unk> if it's length is larger than this
            (defaults to None, which is equal to +infinity)
        token_mask_prob: probability of masking token while training
        provide_subword_tags: output tags for subwords or for words
    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: rmax lenght of a bert subtoken
        tokenizer: instance of Bert FullTokenizer
    """

    def __init__(self,
                 max_seq_length: int = 4096,
                 max_subword_length: int = 15,
                 token_maksing_prob: float = 0.0,
                 provide_subword_tags: bool = False,
                 **kwargs):
        self._re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.provide_subword_tags = provide_subword_tags
        self.mode = kwargs.get('mode')
        self.max_seq_length = max_seq_length
        self.max_subword_length = max_subword_length
        self.tokenizer = FullTokenizer(vocab_file=VOCAB_PATH, do_lower_case=False)
        self.token_maksing_prob = token_maksing_prob

        self.log = getLogger(__name__)

    def __call__(self,
                 tokens: Union[List[List[str]], List[str]],
                 tags: List[List[str]] = None,
                 **kwargs):
        if isinstance(tokens[0], str):
            tokens = [re.findall(self._re_tokenizer, s) for s in tokens]
        subword_tokens, subword_tok_ids, subword_masks, subword_tags = [], [], [], []
        for i in range(len(tokens)):
            toks = tokens[i]
            ys = ['O'] * len(toks) if tags is None else tags[i]
            mask = [int(y != 'X') for y in ys]
            print("toks")
            print(toks)
            print("ys")
            print(ys)
            print("KKKK")
            assert len(toks) == len(ys) == len(mask), \
                f"toks({len(toks)}) should have the same length as " \
                    f" ys({len(ys)}) and mask({len(mask)}), tokens = {toks}."
            sw_toks, sw_mask, sw_ys = self._ner_bert_tokenize(toks,
                                                              mask,
                                                              ys,
                                                              self.tokenizer,
                                                              self.max_subword_length,
                                                              mode=self.mode,
                                                              token_maksing_prob=self.token_maksing_prob)
            if self.max_seq_length is not None:
                if len(sw_toks) > self.max_seq_length:
                    print("sw_toks")
                    print(sw_toks)
                    print(len(sw_toks))
                    raise RuntimeError(f"input sequence after bert tokenization"
                                       f" shouldn't exceed {self.max_seq_length} tokens. {len(sw_toks)} is")
            subword_tokens.append(sw_toks)
            subword_tok_ids.append(self.tokenizer.convert_tokens_to_ids(sw_toks))
            subword_masks.append(sw_mask)
            subword_tags.append(sw_ys)
            assert len(sw_mask) == len(sw_toks) == len(subword_tok_ids[-1]) == len(sw_ys), \
                f"length of mask({len(sw_mask)}), tokens({len(sw_toks)})," \
                    f" token ids({len(subword_tok_ids[-1])}) and ys({len(ys)})" \
                    f" for tokens = `{toks}` should match"
        subword_tok_ids = self.zero_pad(subword_tok_ids, dtype=int, padding=0)
        subword_masks = self.zero_pad(subword_masks, dtype=int, padding=0)
        if tags is not None:
            if self.provide_subword_tags:
                return tokens, subword_tokens, subword_tok_ids, subword_masks, subword_tags
            else:
                nonmasked_tags = [[t for t in ts if t != 'X'] for ts in tags]
                for swts, swids, swms, ts in zip(subword_tokens,
                                                 subword_tok_ids,
                                                 subword_masks,
                                                 nonmasked_tags):
                    if (len(swids) != len(swms)) or (len(ts) != sum(swms)):
                        self.log.warning('Not matching lengths of the tokenization!')
                        self.log.warning(f'Tokens len: {len(swts)}\n Tokens: {swts}')
                        self.log.warning(f'Masks len: {len(swms)}, sum: {sum(swms)}')
                        self.log.warning(f'Masks: {swms}')
                        self.log.warning(f'Tags len: {len(ts)}\n Tags: {ts}')
                return tokens, subword_tokens, subword_tok_ids, subword_masks, nonmasked_tags
        return tokens, subword_tokens, subword_tok_ids, subword_masks

    @staticmethod
    def _ner_bert_tokenize(tokens: List[str],
                           mask: List[int],
                           tags: List[str],
                           tokenizer: FullTokenizer,
                           max_subword_len: int = None,
                           mode: str = None,
                           token_maksing_prob: float = 0.0) -> Tuple[List[str], List[int], List[str]]:
        tokens_subword = ['[CLS]']
        mask_subword = [0]
        tags_subword = ['X']
        for token, flag, tag in zip(tokens, mask, tags):
            subwords = tokenizer.tokenize(token)
            if not subwords or \
                    ((max_subword_len is not None) and (len(subwords) > max_subword_len)):
                tokens_subword.append('[UNK]')
                mask_subword.append(flag)
                tags_subword.append(tag)
            else:
                if mode == 'train' and token_maksing_prob > 0.0 and np.random.rand() < token_maksing_prob:
                    tokens_subword.extend(['[MASK]'] * len(subwords))
                else:
                    tokens_subword.extend(subwords)
                mask_subword.extend([flag] + [0] * (len(subwords) - 1))
                tags_subword.extend([tag] + ['X'] * (len(subwords) - 1))

        tokens_subword.append('[SEP]')
        mask_subword.append(0)
        tags_subword.append('X')

        return tokens_subword, mask_subword, tags_subword

    def zero_pad(self, batch, zp_batch=None, dtype=np.float32, padding=0):
        if zp_batch is None:
            dims = self.get_dimensions(batch)
            zp_batch = np.ones(dims, dtype=dtype) * padding
        if zp_batch.ndim == 1:
            zp_batch[:len(batch)] = batch
        else:
            for b, zp in zip(batch, zp_batch):
                self.zero_pad(b, zp)
        return zp_batch

    def get_dimensions(self, batch) -> List[int]:
        return list(map(max, self.get_all_dimensions(batch)))

    def get_all_dimensions(self, batch: Sequence, level: int = 0, res: Optional[List[List[int]]] = None) -> List[List[int]]:
        if not level:
            res = [[len(batch)]]
        if len(batch) and isinstance(batch[0], Sized) and not isinstance(batch[0], str):
            level += 1
            if len(res) <= level:
                res.append([])
            for item in batch:
                res[level].append(len(item))
                self.get_all_dimensions(item, level, res)
        return res


tokenizer_bert = FullTokenizer(vocab_file=VOCAB_PATH, do_lower_case=False)
_tokenizer = BertNerPreprocessor(max_seq_length=SEQ_LEN, tokenizer = tokenizer_bert)

def tokenize(utterance: List[str], labels: Optional[List[str]]):
    def mask(tokens_batch):
        batch_size = len(tokens_batch)
        max_len = max(len(utt) for utt in tokens_batch)
        mask = np.zeros([batch_size, max_len], dtype=np.int64)
        for n, utterance in enumerate(tokens_batch):
            mask[n, :len(utterance)] = 1

        return mask

    tag_dict = TAGS_MAP

    PAD_ID = tokenizer_bert.convert_tokens_to_ids(['[PAD]'])[0]

    tokens = _tokenizer(utterance, labels)

    subword_ids = tokens[2][0]
    attention_mask = mask(tokens[1])[0]
    first_subword_mask = tokens[3][0]
    for _ in range(len(subword_ids), SEQ_LEN):
        subword_ids = np.append(subword_ids, PAD_ID)
        attention_mask = np.append(attention_mask, 0)
        first_subword_mask = np.append(first_subword_mask, 0)

    if labels:
        tags = [ tag_dict[tag] for tag in tokens[4][0] ]
        for _ in range(len(tags), SEQ_LEN):
            tags.append(OUT_OF_SENTENCE_CODE)
        return subword_ids, attention_mask, first_subword_mask, utterance, tags
    else:
        return subword_ids, attention_mask, first_subword_mask, utterance
