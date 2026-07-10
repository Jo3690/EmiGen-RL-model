import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from hgraph import *
from hgraph import HierVAE, common_atom_vocab, PairVocab


param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))



import sys 
sys.path.append("..")  

from core.transform import SubgraphsTransform
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
import argparse
from torch_geometric.loader import DataLoader as GDataLoader

from transfer_package import create_model,create_qy_model,OledData
import hgraph
from hgraph import *

from torch.distributions import MultivariateNormal

@torch.no_grad()
def predict(loader, model, device, mean, std):
    model.eval()   
    model_output = []
    
    for data in loader:
        data = data.to(device)
        output = model(data).reshape(-1)
        output = output * std + mean
                        
        
        model_output.extend(output.tolist())                                       


    return model_output


data = [line.strip().split('\t') for line in open("./deep4chem_emission.txt").readlines()]  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y = [float(i[2]) for i in data]
Y = np.array([i for i in y])
mean = Y.mean()
std = Y.std()
mean = float(mean)
std = float(std)

qy = [line.strip().split('\t') for line in open("./deep4chem_plqy.txt").readlines()]  

plqy = [float(i[2]) for i in qy]
Y_qy = np.array([i for i in plqy])
mean_qy = Y_qy.mean()
std_qy = Y_qy.std()
mean_qy = float(mean_qy)
std_qy = float(std_qy)

def create_dataset(): 
    transform_eval = SubgraphsTransform(3, 
                                        walk_length=0, 
                                        p=1.0, 
                                        q=1.0, 
                                        repeat=5,
                                        sampling_mode=None, 
                                        random_init=False)


    if os.path.exists(r'./data/raw/data_list.pt'):
        os.remove(r'./data/raw/data_list.pt')
    if os.path.exists(r'./data/full/processed/val.pt'):
        os.remove(r'./data/full/processed/val.pt')
        os.remove(r'./data/full/processed/pre_transform.pt')
        os.remove(r'./data/full/processed/pre_filter.pt')
        
    root = './data'
     
    test_dataset = OledData(root, subset=0, split='val', file_name='samples.txt',test=False,transform=transform_eval,percentage=0.0,mean=mean,std=std) 
    
    test_dataset = [x for x in test_dataset] 

    test_loader = GDataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8,follow_batch=['edge_attr'])

    return test_loader

def qy_dataset(): 
    transform_eval = SubgraphsTransform(3, 
                                        walk_length=0, 
                                        p=1.0, 
                                        q=1.0, 
                                        repeat=5,
                                        sampling_mode=None, 
                                        random_init=False)


    if os.path.exists(r'./qy/raw/data_list.pt'):
        os.remove(r'./qy/raw/data_list.pt')
    if os.path.exists(r'./qy/full/processed/val.pt'):
        os.remove(r'./qy/full/processed/val.pt')
        os.remove(r'./qy/full/processed/pre_transform.pt')
        os.remove(r'./qy/full/processed/pre_filter.pt')
    root = './qy'
     
    test_dataset = OledData(root, subset=0, split='val', file_name='samples.txt',test=False,transform=transform_eval,percentage=0.0,mean=mean,std=std) 
    
    test_dataset = [x for x in test_dataset] 

    test_loader = GDataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8,follow_batch=['edge_attr'])

    return test_loader

def sample_predict(smiles):
    
    with open("./data/raw/samples.txt", 'w') as f:
        for idx,(e, s) in enumerate(smiles):
            if idx+1 == len(smiles):
                f.write(e+'\t'+ s)
            else:
                f.write(e+'\t'+ s + '\n')
                
    with open("./qy/raw/samples.txt", 'w') as f:
        for idx,(e, s) in enumerate(smiles):
            if idx+1 == len(smiles):
                f.write(e+'\t'+ s)
            else:
                f.write(e+'\t'+ s + '\n')


class ActionNetwork(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),

            nn.ReLU(),
            nn.Linear(64, 64),

            nn.ReLU(),
            nn.Linear(64, latent_dim),

            nn.ReLU()
        )
        self.action_var = nn.Parameter(torch.ones(latent_dim)*0.420)
    
    def forward(self, z):
       
        mean = self.fc(z) 
     
        cov_mat = torch.diag_embed(self.action_var).expand(z.size(0), -1, -1)
        dist = MultivariateNormal(mean, cov_mat)
        return dist



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
            nn.ReLU()
        )
        self.actor = ActionNetwork(action_dim)
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)

        )
 
    def forward(self, state):
        shared = self.shared_layer(state)
        action_probs = self.actor(shared)
        state_value = self.critic(shared)
        return action_probs, state_value
    

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
 
    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
import sascorer
class sa_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                scores.append(sascorer.calculateScore(mol))
        return np.float32(scores)

def batch_calc_reward(smiles):

    
    test_loader = create_dataset()
    qy_loader = qy_dataset()
    output = predict(test_loader, prop_predictor, device=device,mean=mean, std=std)
    
    output_qy = predict(qy_loader, qy_predictor, device=device,mean=mean_qy, std=std_qy)
    



    score,score_qy = [],[]
    for i in output:
        if i >= 400 and i <= 500:
            j = (500-i) / 100
        else:
            j = 0
        score.append(j)

    for i in output_qy:
        if i >= 0 and i <= 1:
            j = i
        elif i > 1 and i <= 1.1:
            j = 1
        else:
            j = 0
        score_qy.append(j)

        
    score_tensor = torch.tensor(score, dtype=torch.float32).to(device)
    score_qy_tensor = torch.tensor(score_qy, dtype=torch.float32).to(device)

    

    return (score_tensor+score_qy_tensor)/2, output,output_qy

def is_majority_ones(tensor):

    ones_ratio = torch.sum(tensor == 1.0).item() / tensor.numel()
    return ones_ratio > 0.5


class PPO:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, eps_clip=0.2, K_epochs=10,epsilon=0.2):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
 
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.epsilon = epsilon
        
    def select_action(self, state, memory):

        action_probs, _ = self.policy_old(state)
        dist = action_probs
        action = dist.rsample()

        

 
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
 
        return action
    
    def update(self, memory,save_dir,episode):

        ckpt = (self.policy.state_dict(), self.optimizer.state_dict())
        torch.save(ckpt, os.path.join(save_dir, "Eisode {} PPO_model_random.ckpt".format(episode)))
        old_states = torch.cat(memory.states).to(device).detach()
        old_actions = torch.cat(memory.actions).to(device).detach()
        old_logprobs = torch.cat(memory.logprobs).to(device).detach()



        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for _ in range(self.K_epochs):

            action_probs, state_values = self.policy(old_states)
            dist = action_probs
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()


            ratios = torch.exp(new_logprobs - old_logprobs.detach())


            advantages = rewards - state_values.detach().squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()


            loss_critic = self.MseLoss(state_values.reshape(-1), rewards.reshape(-1))
 

            loss = loss_actor + 0.5 * loss_critic - 0.05 * entropy.mean()
            print("Total Loss:",loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


    def StateAction(self, state, action,t):



        state = state + self.epsilon * action
        

        smi_list = model.decoder.decode((state, state, state), greedy=0, max_decode_step=150)
        smiles = [[] for i in range(len(condsol))]
        
        for idx,j in enumerate(smi_list):
            smiles[idx].append(j)
            smiles[idx].append(condsol[idx])


        sample_predict(smiles)
        reward,emi_pred,plqy_pred = batch_calc_reward(smiles) 
        print("time_step:",t, "reward:",reward,"mean_reward",torch.mean(reward))
        done = is_majority_ones(reward)
        rewarded_smiles = []
        for i, r in enumerate(reward):
            generated_smiles = smiles[i][0]
            rewarded_smiles.append((generated_smiles,emi_pred[i],plqy_pred[i], float(r)))
        return state, reward, done,rewarded_smiles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_path = './emiModelParams.pkl'
qy_path = './qyModelParams.pkl'

predict_model = create_model().to(device)
predict_model.load_state_dict(torch.load(pred_path),strict = False)
prop_predictor = predict_model.to(device)

qy_model = create_qy_model().to(device)
qy_model.load_state_dict(torch.load(qy_path),strict = False)
qy_predictor = qy_model.to(device)

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--generative_model', required=True)
    
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--file_name', type=str, default='rl_two.txt')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=1)


    args = parser.parse_args()
    print(args)




    condition_dict = {"solvent":"CC1=CC=CC=C1"}

    
    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
    
    args.vocab = PairVocab([(x,y) for x,y in vocab])



    model = HierVAE(args).cuda()


    print('Loading from checkpoint ' + args.generative_model)

    model_state = torch.load(args.generative_model)
    model.load_state_dict(model_state,strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    a = 0
    condsol = [condition_dict["solvent"] for i in range(args.batch_size)]


    


    state_dim = args.latent_size
    action_dim = args.latent_size
    lr = 0.001
    gamma = 0.99
    eps_clip = 0.1
    K_epochs = 200
    max_episodes = 100
    max_timesteps = 5
    

    ppo = PPO(state_dim, action_dim, lr, gamma, eps_clip, K_epochs)
    memory = Memory()


    file_path = args.file_name


    if os.path.exists(file_path):
        os.remove(file_path)
    for episode in range(1, max_episodes + 1):
        state = model.sample_z(args.batch_size)
        total_reward = 0
        for t in range(max_timesteps):
            
            action = ppo.select_action(state, memory)
            state, reward, done, reward_smi = ppo.StateAction(state, action,t)
            with open(file_path,"a") as txt:
                txt.write("episode{} timestep {}".format(episode,t)+"\n")
                for i in reward_smi:
                    txt.write(i[0]+"\t"+str(i[1])+"\t"+str(i[2])+"\t"+str(i[3])+"\n")
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            total_reward += reward
            
            if done:
                break
    
        ppo.update(memory,args.save_dir,episode)
        memory.clear()
    
        print(f"Episode {episode}, Total Reward: {total_reward}")