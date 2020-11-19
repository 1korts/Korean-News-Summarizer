import torch

import pandas as pd
import re

from utils.tokenization_kobert import KoBertTokenizer

"""
Codes for transform raw data into usable data.
Raw data used in this project are from "AI HUB, Korea"

We used
    1400 News data for the training,
    400 News data for the validation,
    200 News data for the test.

The data are preprocessed to be suitable for "KoBERT", the encoder used in this project.

The tokenizer used in this project is borrowed from "github.com/monologg"
"""


def load_excel_to_pt(path, tokenizer, doc_max_len, tgt_max_len, save_path):

    df = pd.read_excel(path)
    sub_newline = re.compile("[\n]+")
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    dataset = []

    for i in range(df.shape[0]):
        ex = df.iloc[i]
        src_txt = str(ex["내용"])
        tgt_txt = str(ex["생성"])

        ext1 = str(ex["추출1"]).strip()
        ext2 = str(ex["추출2"]).strip()
        ext3 = str(ex["추출3"]).strip()

        ext = [ext1, ext2, ext3]

        src_txt = sub_newline.sub("\n", src_txt.strip())
        src_txt_split = src_txt.split("\n")

        ext_sent = []

        for j, s in enumerate(src_txt_split):
            if s in ext:
                ext_sent.append(j)

        text = ' [SEP] [CLS] '.join(src_txt_split)
        src_tokens = tokenizer.tokenize(text)[:doc_max_len - 2]
        src_tokens = ['[CLS]'] + src_tokens + ['[SEP]']

        src_token_ids = tokenizer.convert_tokens_to_ids(src_tokens)

        _segs = [-1] + [i for i, t in enumerate(src_token_ids) if t == sep_id]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []

        for j, s in enumerate(segs):
            if j % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        cls_ids = [i for i, t in enumerate(src_token_ids) if t == cls_id]

        sent_labels = [0] * len(cls_ids)
        for j in range(len(sent_labels)):
            if j in ext_sent:
                sent_labels[j] = 1

        tgt_txt = sub_newline.sub("\n", tgt_txt.strip())
        tgt_tokens = tokenizer.tokenize(tgt_txt)[:tgt_max_len - 2]
        tgt_tokens = ['[CLS]'] + tgt_tokens + ['[SEP]']
        tgt_token_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)

        data = {
            "src": src_token_ids,
            "tgt": tgt_token_ids,
            "segments_ids": segments_ids,
            "cls_ids": cls_ids,
            "sent_labels": sent_labels
        }

        dataset.append(data)

    train_dataset = dataset[:1400]
    validation_dataset = dataset[1400:1800]
    test_data_set = dataset[1800:]

    datasets = [train_dataset, validation_dataset, test_data_set]
    for s, d in zip(save_path, datasets):
        torch.save(d, s)


if __name__ == "__main__":

    MAX_SRC_TOKENS = 512
    MAX_TGT_TOKENS = 140

    RAW_DATA_PATH = "storage/raw_data/data.xlsx"
    TRAIN_DATA_PATH = "storage/data/train_data.pt"
    VALIDATION_DATA_PATH = "storage/data/validation_data.pt"
    TEST_DATA_PATH = "storage/data/test_data.pt"

    PATHS = [TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH]
    TOKENIZER = KoBertTokenizer.from_pretrained("monologg/kobert")
    load_excel_to_pt(RAW_DATA_PATH, TOKENIZER, MAX_SRC_TOKENS, MAX_TGT_TOKENS, PATHS)