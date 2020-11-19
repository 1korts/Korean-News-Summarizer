from abs_summarizer.model import AbsSummarizer
from abs_summarizer.optimizer import build_optim_bert, build_optim_dec
from common.data_loader import Dataset
import torch
from utils.tokenization_kobert import KoBertTokenizer
from abs_summarizer.loss import abs_loss
from abs_summarizer.train_helper import Statistics
from ext_summarizer.model import ExtSummarizer

from tqdm import tqdm

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# params for the model
params = {
    "bert_fine_tune": True,
    "use_bert_emb": True,
    "dec_layers": 6,
    "dec_hidden_size": 768,
    "dec_ff_size": 2048,
    "dec_dropout": 0.1,
    "label_smoothing": 0.1,
    "generator_shard_size": 5,
    "dec_heads": 12
}

ext_params = {
    "bert_fine_tune": True,
    "ext_ff": 2048,
    "ext_heads": 8,
    "ext_dropout": 0.1,
    "ext_layers": 2,
    "hidden_size": 768
}

# params for the optimizer
optim_params = {
    "use_gpu": True,
    "method": "adam",
    "lr_bert": 2e-3,
    "lr_dec": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "warmup_steps_bert": 20000,
    "warmup_steps_dec": 10000,
    "max_grad_norm": 0
}
# checkpoint of the model
checkpoint = None
extractive_path = "storage/ext_trained_model/model.pt"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if extractive_path is not None:
    print("Start loading the BERT model.")
    checkpoint_ext = torch.load(extractive_path, torch.device('cpu'))
    pre_trained_extractive_model = ExtSummarizer(ext_params, device, checkpoint=checkpoint_ext)
    bert_from_extractive = pre_trained_extractive_model.bert
    print("BERT model is loaded.")
else:
    bert_from_extractive = None

# training options
batch_size = 2
train_steps = 5
validate_each = 1

if __name__ == "__main__":
    model = AbsSummarizer(params=params, device=device, checkpoint=checkpoint,
                          bert_from_extractive=bert_from_extractive)
    optim_bert = build_optim_bert(optim_params, model, checkpoint)
    optim_dec = build_optim_dec(optim_params, model, checkpoint)
    optims = [optim_bert, optim_dec]

    train_dataset = Dataset("storage/data/train_data.pt")
    valid_dataset = Dataset("storage/data/validation_data.pt")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                               collate_fn=train_dataset.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                               collate_fn=valid_dataset.collate_fn)

    tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    symbols = {'BOS': 2, 'EOS': 3,
               'PAD': 1, 'EOQ': 0}

    criterion = abs_loss(model.generator, symbols, model.vocab_size, device, train=True,
                         label_smoothing=params["label_smoothing"])

    step = 0
    best_loss = None

    while step < train_steps:

        stats = Statistics()
        model.train()
        for batch in tqdm(train_loader):
            batch.to(device)
            model.zero_grad()

            src = batch.src
            tgt = batch.tgt
            segs = batch.segs
            clss = batch.clss
            mask_src = batch.mask_src
            mask_tgt = batch.mask_tgt
            mask_cls = batch.mask_clss

            outputs, _ = model(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)
            num_tokens = batch.tgt[:, 1:].ne(criterion.padding_idx).sum()
            normalization = num_tokens.item()
            batch_stats = criterion.sharded_compute_loss(batch, outputs, 32, normalization)

            stats.update(batch_stats)

            optims[0].step()
            optims[1].step()

        print("==== Epoch : %d, Train Loss : %.2f, Train Accuracy: %.2f" % (step+1, stats.loss / len(train_loader),
                                                                            stats.accuracy()))
        if (step + 1) % validate_each == 0:
            with torch.no_grad():
                valid_loss = 0
                stats = Statistics()
                model.eval()
                for batch in tqdm(valid_loader):
                    batch.to(device)

                    src = batch.src
                    tgt = batch.tgt
                    segs = batch.segs
                    clss = batch.clss
                    mask_src = batch.mask_src
                    mask_tgt = batch.mask_tgt
                    mask_cls = batch.mask_clss

                    outputs, _ = model(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)
                    num_tokens = batch.tgt[:, 1:].ne(criterion.padding_idx).sum()
                    normalization = num_tokens.item()
                    batch_stats = criterion.monolithic_compute_loss(batch, outputs)

                    stats.update(batch_stats)

                print("==== Epoch : %d, Valid Loss : %.2f, Valid Accuracy: %.2f" % (step+1,
                                                                                    stats.loss / len(valid_loader),
                                                                                    stats.accuracy()))

                if best_loss is None:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'optims': optims
                    }
                    checkpoint_path = os.path.join("storage/abs_trained_model", "model.pt")
                    torch.save(checkpoint, checkpoint_path)
                    best_loss = valid_loss
                elif valid_loss < best_loss:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'optims': optims
                    }
                    checkpoint_path = os.path.join("storage/abs_trained_model", "model.pt")
                    torch.save(checkpoint, checkpoint_path)
                    best_loss = valid_loss
        step += 1
