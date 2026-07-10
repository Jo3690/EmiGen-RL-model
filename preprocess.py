from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import os

from hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ncpu', type=int, default=64)
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    args.vocab = PairVocab(vocab, cuda=False)

    pool = Pool(args.ncpu) 
    random.seed(42)

    with open(args.train) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    random.shuffle(data)
    batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
    print("start")
    all_data = []
    for i in batches:
        d = tensorize(i,vocab = args.vocab)
        all_data.append(d)           

    print("done")
    num_splits = len(all_data) // 1000
    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

            
        save_path = os.path.join(args.save_dir, 'tensors-%d.pkl' % split_id)
        with open(save_path, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)