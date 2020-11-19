import torch

from common.data_loader import Dataset
from ext_summarizer.model import ExtSummarizer
from ext_summarizer.optimizer import build_optim
from utils.evaluation import prediction, add_count, f1_score

from tqdm import tqdm

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# params for the model
params = {
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
    "lr": 2e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "warmup_steps": 10000,
    "max_grad_norm": 0
}

# checkpoint of the model
checkpoint = None

# training options
batch_size = 4
train_steps = 5
num_workers = 4
validate_each = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    model = ExtSummarizer(params=params, device=device, checkpoint=checkpoint)
    optim = build_optim(optim_params, model, checkpoint)
    criterion = torch.nn.BCELoss(reduction='none')

    train_dataset = Dataset("storage/data/train_data.pt")
    valid_dataset = Dataset("storage/data/validation_data.pt")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, collate_fn=train_dataset.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, collate_fn=valid_dataset.collate_fn)

    step = 0
    best_loss = None

    while step < train_steps:
        train_loss = 0
        train_count = [0, 0, 0, 0]
        for batch in tqdm(train_loader):
            batch.to(device)
            model.zero_grad()

            src = batch.src
            labels = batch.sent_labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_clss = batch.mask_clss

            sent_scores, mask = model(src, segs, clss, mask, mask_clss)
            loss = criterion(sent_scores, labels.float())
            loss = (loss * mask.float()).sum()
            (loss / loss.numel()).backward()

            train_loss += (loss / loss.numel()).item()

            pred = prediction(sent_scores, 3).to(device)
            train_count = add_count(train_count, pred, labels)
            optim.step()

        print("==== Epoch : %d, Train Loss : %.2f, Train F1 Score: %.3f" % (step+1, train_loss / len(train_loader),
                                                                            f1_score(train_count)))

        if (step + 1) % validate_each == 0:
            with torch.no_grad():
                model.eval()
                valid_loss = 0
                valid_count = [0, 0, 0, 0]
                for batch in tqdm(valid_loader):
                    batch.to(device)

                    src = batch.src
                    labels = batch.sent_labels
                    segs = batch.segs
                    clss = batch.clss
                    mask = batch.mask_src
                    mask_clss = batch.mask_clss

                    sent_scores, mask = model(src, segs, clss, mask, mask_clss)
                    loss = criterion(sent_scores, labels.float())
                    loss = (loss * mask.float()).sum()

                    valid_loss += (loss / loss.numel()).item()
                    pred = prediction(sent_scores, 3).to(device)
                    valid_count = add_count(valid_count, pred, labels)

                print(
                    "==== Epoch : %d, Valid Loss : %.2f, Valid F1 Score: %.3f" % (step+1, valid_loss / len(valid_loader),
                                                                                  f1_score(valid_count)))

                if best_loss is None:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'optim': optim
                    }
                    checkpoint_path = os.path.join("storage/ext_trained_model", "model.pt")
                    torch.save(checkpoint, checkpoint_path)
                    best_loss = valid_loss
                elif valid_loss < best_loss:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'optim': optim
                    }
                    checkpoint_path = os.path.join("storage/ext_trained_model", "model.pt")
                    torch.save(checkpoint, checkpoint_path)
                    best_loss = valid_loss

        step += 1
