import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
import datetime
from hgraph import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--beta', type=float, default=0.00)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=10.0)
parser.add_argument('--step_beta', type=float, default=0.005)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=7000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=1)


args = parser.parse_args()

log_path = os.path.join(args.save_dir, "train_log.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
open(log_path, 'w').close()

os.makedirs(os.path.dirname("./nohup.out"), exist_ok=True)
open("./nohup.out", 'w').close()


start_time_obj = datetime.datetime.now()
with open(log_path, 'a') as f:
    f.write(f"[INFO] Training started at: {start_time_obj.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")


with open(log_path, 'a') as f:
    f.write(str(args) + '\n')
torch.manual_seed(args.seed)
random.seed(args.seed)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()
#params_nums = str("Model
#with open(log_path, 'a') as f:
    #f.write(params_nums + '\n')

    

if args.load_model:
    from_ckpt = 'continuing from checkpoint ' + args.load_model
    with open(log_path, 'a') as f:
        f.write(from_ckpt + '\n')
    ckpoint = torch.load(args.load_model)
    total_step = 0
    beta = args.beta
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    model.load_state_dict(ckpoint["model_state_dict"],strict=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(ckpoint['optimizer_state_dict'])



else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)        
    total_step = 0
    beta = args.beta

from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=args.anneal_rate,
    patience=1000,
    threshold=1e-4,
    min_lr=1e-7,
    verbose=True
)


param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

meters = np.zeros(6)
best_avg_acc = -1
dataset = DataFolder(args.train, args.batch_size)

for epoch in range(args.epoch):
    for batch in tqdm(dataset):
        total_step += 1
        model.zero_grad()
        loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([
                                    kl_div,
                                    loss.item(),
                                    wacc.cpu().item() * 100,
                                    iacc.cpu().item() * 100,
                                    tacc.cpu().item() * 100,
                                    sacc.cpu().item() * 100
                                ])
        bacteria = float(np.array([ wacc.cpu().item() * 100,
                                    iacc.cpu().item() * 100,
                                    tacc.cpu().item() * 100,
                                    sacc.cpu().item() * 100]).mean())
        
        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            scheduler.step(bacteria)
            log_str = "Epoch [%d] Step [%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (epoch, total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model))
            with open(log_path, 'a') as f:
                f.write(log_str + '\n')

            meters *= 0

        
        if bacteria > best_avg_acc:
            best_avg_acc = bacteria
            if args.load_model is None:
                ckpt_path = os.path.join(args.save_dir, f"{epoch}_model.ckpt")
            else:
                ckpt_path = os.path.join(args.save_dir, f"{epoch}_model_from_load.ckpt")
            checkpoint = {
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'total_step': total_step,
                                'beta': beta
                            }
            torch.save(checkpoint, ckpt_path)
            save_msg = f"{epoch} epoch {total_step} step with the New best model saved with avg acc: {best_avg_acc:.2f} with word: {float(wacc):.2f} and assemble {float(sacc):.2f}"
            
            current_lr = optimizer.param_groups[0]['lr']

            lr_msg  = f"Actual LR: {current_lr:.2e}"

            with open(log_path, 'a') as f:
                f.write(save_msg + '\n')
                f.write(lr_msg + '\n')
            
        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)
            
end_time_obj = datetime.datetime.now()
elapsed = end_time_obj - start_time_obj
total_sec = elapsed.total_seconds()
hours, remainder = divmod(total_sec, 3600)
minutes, seconds = divmod(remainder, 60)

with open(log_path, 'a') as f:
    f.write("\n" + "="*60 + "\n")
    f.write(f"[INFO] Training finished at: {end_time_obj.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"[INFO] Total training time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s\n")