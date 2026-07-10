import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import math, random, sys
import numpy as np
import argparse
import hgraph
from hgraph import *
import rdkit
from rdkit import Chem
import os
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--generative_model', required=True)
parser.add_argument('--load_epoch', type=int, default=-1)


parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--clip_norm', type=float, default=10.0)
parser.add_argument('--beta', type=float, default=0.1)

parser.add_argument('--inner_epoch', type=int, default=500)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=-1)

args = parser.parse_args()
print(args)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
args.vocab = PairVocab([(x,y) for x,y,_ in vocab])

model = HierVAE(args).cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

print('Loading from checkpoint ' + args.generative_model)
model_state = torch.load(args.generative_model)
model.load_state_dict(model_state,strict=False)
optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=args.anneal_rate,
    patience=3,
    threshold=1e-4,
    min_lr=1e-7,
    verbose=True
)
train_data = [line.strip().split('\t') for line in open(args.train).readlines()]
train_smiles = []
for i in train_data:
    m = Chem.MolFromSmiles(i[0])
    smi = Chem.MolToSmiles(m)
    train_smiles.append(smi)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = args.beta
total_rate = 0
meters = np.zeros(6)

print("start**************",len(train_smiles))
for epoch in range(args.epoch):

    
    dataset = hgraph.MoleculeDataset(train_smiles, args.vocab, args.atom_vocab, args.batch_size)
    a = 0
    

    print(f'Epoch {epoch} training...')
    for epo in range(args.inner_epoch):
        meters = np.zeros(6)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x:x[0], shuffle=0, num_workers=0)
        
        for batch in tqdm(dataloader):
            model.zero_grad()

            
            loss, kl_div, wacc, iacc, tacc, sacc = model(*batch[0], beta=beta)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])
            bacteria = float(np.array([wacc * 100, iacc * 100, tacc * 100, sacc * 100]).mean())
        a += 1
        meters /= len(dataset)
        scheduler.step(bacteria)
        if bacteria > total_rate:

            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{epo}_FineTune_model.ckpt"))
            total_rate = bacteria
        print("Epoch:%i Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (a, beta, meters[0], meters[1], wacc*100, iacc*100, tacc*100, sacc*100, param_norm(model), grad_norm(model)))
