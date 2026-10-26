"""Exercise FAST IDs and missing-frame slots through released tokenization/noising."""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from native_recovery_common import released, read, write
from fast_recovery_common import Corpus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vocabulary-size', type=int, choices=(4096, 8192, 16384), default=4096)
    a = p.parse_args()
    torch.set_num_threads(2)
    root = a.native / 'candidates/v2'
    cfg = read(root / 'config.json')
    old_vocab = Path(cfg['vocab_directory'])
    vocab = (old_vocab / 'vocab.txt').read_text().splitlines()[:36]
    size = a.vocabulary_size
    width = max(4, len(str(size-1)))
    codes = [f'{i:0{width}d}' for i in range(size)]
    vocab += codes + ['<mask_struct>']
    api = released(root)
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        (temp / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
        token_cfg = read(old_vocab / 'tokenizer_config.json')
        token_cfg['added_tokens_decoder'] = {str(vocab.index(item['content'])): item
            for item in token_cfg['added_tokens_decoder'].values()}
        write(temp / 'tokenizer_config.json', token_cfg)
        write(temp / 'special_tokens_map.json', read(old_vocab / 'special_tokens_map.json'))
        tokenizer = api.tokenizer_class.from_pretrained(temp)
        assert len(tokenizer) == size+37 and tokenizer.all_tokens == vocab
        assert tokenizer.convert_tokens_to_ids('<mask_struct>') == size+36
        assert tokenizer.convert_tokens_to_ids(codes[-1]) == size+35
        # Test every code plus shuffled adjacencies through the actual tokenizer,
        # including the decimal boundary at 9999/10000 for the 16k vocabulary.
        order = np.random.default_rng(42).permutation(size).tolist()
        assert tokenizer.encode(''.join(codes), add_special_tokens=False) == list(range(36, size+36))
        assert tokenizer.encode(''.join(codes[i] for i in order), add_special_tokens=False) == [36+i for i in order]
        cache = temp / 'cache'
        cache.mkdir()
        np.save(cache / 'aa.npy', np.array([vocab.index(c) for c in 'ACDEF'], dtype=np.uint8))
        np.save(cache / 'codes.npy', np.array([0, 65535, size-1, 23, 4], dtype=np.uint16))
        np.save(cache / 'structure_mask.npy', np.array([True, False, True, True, True]))
        np.save(cache / 'offsets.npy', np.array([0, 5]))
        (cache / 'rows.jsonl').write_text(json.dumps(dict(id='example', length=5)) + '\n')
        manifest = dict(schema='fast-released-projection-v1', vocabulary_size=size, entries=1, files={})
        write(cache / 'manifest.json', manifest)
        corpus = Corpus(cache, tokenizer.all_tokens, verify_files=False)
        row = corpus.record(0)
        assert row['aa_seq'] == 'ACDEF' and row['length'] == 5
        raw = dict(struct_tokens='<cls_struct>' + row['struct_seq'].replace(',', '') + '<eos_struct>',
                   aatype_tokens='<cls_aa>' + row['aa_seq'] + '<eos_aa>')
        batch = api.collater(tokenizer)([raw])
        st, aa = batch['struct_tokens']['targets'], batch['aatype_tokens']['targets']
        assert st.shape == aa.shape == (1, 7)
        assert st.tolist() == [[33, 36, 1, size+35, 59, 40, 34]]
        net = torch.nn.Module()
        net.config = SimpleNamespace(dplm_type='dplm2')
        model = api.model_class(dict(tokenizer=dict(vocab_file=str(temp)), gradient_ckpt=False), net=net)
        mask = model.get_non_special_symbol_mask(st)
        assert mask.tolist() == [[False, True, False, True, True, True, False]]
        noisy, loss_mask = model.q_sample(st, torch.tensor([500]), model.get_modality_type(st), mask)
        assert noisy.tolist() == [[33, size+36, 1, size+36, size+36, size+36, 34]]
        assert torch.equal(loss_mask, mask)
        assert model.get_non_special_symbol_mask(aa).sum() == 5
        # Zero is a real code. Out-of-range codes and mislabeled native caches
        # must fail, even if a caller bypasses file-hash verification for a test.
        for bad in (size, 65535):
            np.save(cache / 'codes.npy', np.array([bad, 65535, size-1, 23, 4], dtype=np.uint16))
            try:
                Corpus(cache, vocab, verify_files=False)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid valid-slot code accepted')
        write(cache / 'manifest.json', manifest | {'schema': 'released-dplm-token-projection-v1'})
        try:
            Corpus(cache, vocab, verify_files=False)
        except ValueError:
            pass
        else:
            raise AssertionError('native cache relabeling accepted')
    write(a.out, dict(passed=True, released_order_fast_vocabulary=size+37,
        every_numeric_code_tokenized=True, shuffled_code_adjacency_verified=True,
        pad_structure_slot_excluded_from_released_noising_and_loss_mask=True,
        corresponding_aa_slot_retained=True, code_zero_is_valid=True,
        out_of_range_and_native_schema_rejected=True,
        scope='CPU tokenizer/data/noising contract; no generator optimization or full-atom quality claim'))


if __name__ == '__main__':
    main()
