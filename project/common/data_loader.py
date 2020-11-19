import torch


def pad_sequence(data: list, pad_id=1):
    width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


class Batch:
    def __init__(self, data=None):
        if data is not None:
            self.src = [d['src'] for d in data]
            self.tgt = [d['tgt'] for d in data]
            self.segs = [d['segments_ids'] for d in data]
            self.clss = [d['cls_ids'] for d in data]
            self.sent_labels = [d['sent_labels'] for d in data]

            self.src = torch.tensor(pad_sequence(self.src))
            self.tgt = torch.tensor(pad_sequence(self.tgt))
            self.segs = torch.tensor(pad_sequence(self.segs, 0))
            self.clss = torch.tensor(pad_sequence(self.clss, -1))
            self.sent_labels = torch.tensor(pad_sequence(self.sent_labels, 0))

            self.mask_src = ~(self.src == 1)
            self.mask_tgt = ~(self.tgt == 1)
            self.mask_clss = ~(self.clss == -1)

    def to(self, device):
        self.src = self.src.to(device)
        self.tgt = self.tgt.to(device)
        self.segs = self.segs.to(device)
        self.mask_src = self.mask_src.to(device)
        self.mask_tgt = self.mask_tgt.to(device)
        self.mask_clss = self.mask_clss.to(device)
        self.sent_labels = self.sent_labels.to(device)

    def __len__(self):
        return len(self.src)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, samples):
        batch = Batch(samples)
        return batch