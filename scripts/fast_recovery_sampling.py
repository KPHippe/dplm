"""Released sampling with strict checkpoint-specific FAST vocabulary checks."""
import numpy as np
import torch
from types import FunctionType
from released_dplm2_control import official as released_official, ALPHABET
from fast_recovery_common import vocabulary_size


def input_tokens(model, record):
    aa = [model.tokenizer._token_to_id[ALPHABET[int(x)]] for x in record['aa']]
    struct = [model.struct_bos_id] + [36]*len(aa) + [model.struct_eos_id]
    return torch.tensor([struct + [model.aa_bos_id] + aa + [model.aa_eos_id]], device=model.device)


def unpack(model, tokens, record, conditional):
    size = vocabulary_size(model.tokenizer.all_tokens)
    n = len(record['aa'])
    if tokens.shape != (1, 2*(n+2)):
        raise ValueError('wrong released output shape')
    st, aa = tokens[0].chunk(2)
    if [int(st[0]), int(st[-1]), int(aa[0]), int(aa[-1])] != [model.struct_bos_id, model.struct_eos_id, model.aa_bos_id, model.aa_eos_id]:
        raise ValueError('released output changed boundary tokens')
    if ((st[1:-1] < 36) | (st[1:-1] >= 36+size)).any():
        raise ValueError('generated code outside FAST vocabulary')
    vocab = model.tokenizer.all_tokens
    sequence = [vocab[int(x)] for x in aa[1:-1]]
    if any(s not in ALPHABET or len(s) != 1 for s in sequence):
        raise ValueError('released output contains noncanonical sequence symbols')
    result = dict(aa=np.array([ALPHABET.index(s) for s in sequence], dtype=np.uint8),
                  fast_id=np.array([int(vocab[int(x)]) for x in st[1:-1]], dtype=np.uint16),
                  residue_id=record['residue_id'], chain_end=record['chain_end'])
    if conditional and not np.array_equal(result['aa'], record['aa']):
        raise ValueError('released generation changed conditioning sequence')
    return result


# Execute the original sampling function unchanged, substituting only numeric
# input/output conversion. Its original parser was explicitly bounded to LFQ8k.
_official = FunctionType(released_official.__code__,
    dict(released_official.__globals__, input_tokens=input_tokens, unpack=unpack),
    released_official.__name__, released_official.__defaults__, released_official.__closure__)


def official(model, record, conditional, steps, seed):
    size = vocabulary_size(model.tokenizer.all_tokens)
    result = _official(model, record, conditional, steps, seed)
    if np.any(result['fast_id'] < 0) or np.any(result['fast_id'] >= size):
        raise ValueError('generated code outside FAST vocabulary')
    return result
