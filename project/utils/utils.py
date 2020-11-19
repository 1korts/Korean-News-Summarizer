import torch
from utils.tokenization_kobert import KoBertTokenizer
from tqdm import tqdm
from common.data_loader import Batch

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def load_text(texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_pos = 512
    tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    sep_vid = tokenizer.convert_tokens_to_ids('[SEP]')
    cls_vid = tokenizer.convert_tokens_to_ids('[CLS]')
    n_lines = len(texts)

    def _process_src(raw):
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + \
            [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(
            src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0

        return src, mask_src, segments_ids, clss, mask_cls

    for x in tqdm(texts, total=n_lines):
        src, mask_src, segments_ids, clss, mask_cls = _process_src(x)
        segs = torch.tensor(segments_ids)[None, :].to(device)
        batch = Batch()
        batch.src = src
        batch.tgt = None
        batch.mask_src = mask_src
        batch.mask_tgt = None
        batch.segs = segs
        batch.src_str = [[sent.replace('[SEP]', '').strip()
                          for sent in x.split('[CLS]')]]
        batch.tgt_str = ['']
        batch.clss = clss
        batch.mask_clss = mask_cls
        batch.batch_size = 1
        yield batch