import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from multigraph.nnutils import *
from multigraph.mol_graph import MolGraph
from multigraph.rnn import GRU, LSTM

class MPNEncoder(nn.Module):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout):
        super(MPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.W_o = nn.Sequential( 
                nn.Linear(node_fdim + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        if rnn_type == 'GRU':
            self.rnn = GRU(input_size, hidden_size, depth) 
        elif rnn_type == 'LSTM':
            self.rnn = LSTM(input_size, hidden_size, depth) 
        else:
            raise ValueError('unsupported rnn cell type ' + rnn_type)

    def forward(self, fnode, fmess, agraph, bgraph):
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
        mask[0, 0] = 0
        return node_hiddens * mask, h

class MultiMPNEncoder(nn.Module):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(MultiMPNEncoder, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.atom_size = atom_size = avocab.size()
        self.bond_size = bond_size = len(MolGraph.BOND_LIST) + MolGraph.MAX_POS

        self.E_c = nn.Sequential(
                nn.Embedding(vocab.size()[0], embed_size),
                nn.Dropout(dropout)
        )
        self.E_i = nn.Sequential(
                nn.Embedding(vocab.size()[1], embed_size),
                nn.Dropout(dropout)
        )
        self.W_c = nn.Sequential( 
                nn.Linear(embed_size + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )
        self.W_i = nn.Sequential( 
                nn.Linear(embed_size + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        self.E_a = torch.eye(atom_size).cuda()
        self.E_b = torch.eye( len(MolGraph.BOND_LIST) ).cuda()
        self.E_apos = torch.eye( MolGraph.MAX_POS ).cuda()
        self.E_pos = torch.eye( MolGraph.MAX_POS ).cuda()

        self.W_root = nn.Sequential( 
                nn.Linear(hidden_size * 2, hidden_size), 
                nn.Tanh()
        )
        self.tree_encoder = MPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.inter_encoder = MPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = MPNEncoder(rnn_type, atom_size + bond_size, atom_size, hidden_size, depthG, dropout)

    def tie_embedding(self, other):
        self.E_c, self.E_i = other.E_c, other.E_i
        self.E_a, self.E_b = other.E_a, other.E_b
    
    def embed_inter(self, tree_tensors, hatom):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput = self.E_i(fnode[:, 1])

        mnode = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        mnode = self.W_i( torch.cat([finput, mnode], dim=-1) )

        hmess = mnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        return mnode, hmess, agraph, bgraph

    def embed_tree(self, tree_tensors, minter):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput = self.E_c(fnode[:, 0])
        mnode = self.W_c( torch.cat([finput, minter], dim=-1) )

        hmess = mnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        return mnode, hmess, agraph, bgraph
    
    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        mnode = self.E_a.index_select(index=fnode, dim=0)
        fmess1 = mnode.index_select(index=fmess[:, 0], dim=0)
        fmess2 = self.E_b.index_select(index=fmess[:, 2], dim=0)
        fpos = self.E_apos.index_select(index=fmess[:, 3], dim=0)
        hmess = torch.cat([fmess1, fmess2, fpos], dim=-1)
        return mnode, hmess, agraph, bgraph

    def embed_root(self, hmess, tree_tensors, roots):
        roots = tree_tensors[2].new_tensor(roots) 
        fnode = tree_tensors[0].index_select(0, roots)
        agraph = tree_tensors[2].index_select(0, roots)

        nei_message = index_select_ND(hmess, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        return self.W_root(node_hiddens)

    def forward(self, tree_tensors, graph_tensors):
        tensors = self.embed_graph(graph_tensors)
        hatom,_ = self.graph_encoder(*tensors)

        tensors = self.embed_inter(tree_tensors, hatom)
        minter,_ = self.inter_encoder(*tensors)

        tensors = self.embed_tree(tree_tensors, minter)
        mnode,hmess = self.tree_encoder(*tensors)
        hroot = self.embed_root(hmess, tensors, [st for st,le in tree_tensors[-1]])

        return hroot, mnode, minter, hatom

class IncMPNEncoder(MPNEncoder):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout):
        super(IncMPNEncoder, self).__init__(rnn_type, input_size, node_fdim, hidden_size, depth, dropout)

    def forward(self, tensors, h, num_nodes, subset):
        fnode, fmess, agraph, bgraph = tensors
        subnode, submess = subset

        if len(submess) > 0: 
            h = self.rnn.sparse_forward(h, fmess, submess, bgraph)

        nei_message = index_select_ND(self.rnn.get_hidden_state(h), 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
        node_hiddens = index_scatter(node_hiddens, node_buf, subnode)
        return node_hiddens, h

class IncMultiMPNEncoder(MultiMPNEncoder):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(IncMultiMPNEncoder, self).__init__(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.tree_encoder = IncMPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.inter_encoder = IncMPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = IncMPNEncoder(rnn_type, self.atom_size + self.bond_size, self.atom_size, hidden_size, depthG, dropout)
        del self.W_root

    def get_sub_tensor(self, tensors, subset):
        subnode, submess = subset
        fnode, fmess, agraph, bgraph = tensors[:4]
        fnode, fmess = fnode.index_select(0, subnode), fmess.index_select(0, submess)
        agraph, bgraph = agraph.index_select(0, subnode), bgraph.index_select(0, submess)

        if len(tensors) == 6:
            cgraph = tensors[4].index_select(0, subnode)
            return fnode, fmess, agraph, bgraph, cgraph, tensors[-1]
        else:
            return fnode, fmess, agraph, bgraph, tensors[-1]

    def embed_sub_tree(self, tree_tensors, hinput, subtree, is_inter_layer):
        subnode, submess = subtree
        num_nodes = tree_tensors[0].size(0)
        fnode, fmess, agraph, bgraph, cgraph, _ = self.get_sub_tensor(tree_tensors, subtree)

        if is_inter_layer:
            finput = self.E_i(fnode[:, 1])
            hinput = index_select_ND(hinput, 0, cgraph).sum(dim=1)
            mnode = self.W_i( torch.cat([finput, hinput], dim=-1) )
        else:
            finput = self.E_c(fnode[:, 0])
            hinput = hinput.index_select(0, subnode)
            mnode = self.W_c( torch.cat([finput, hinput], dim=-1) )

        if len(submess) == 0:
            hmess = fmess
        else:
            node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
            node_buf = index_scatter(mnode, node_buf, subnode)
            hmess = node_buf.index_select(index=fmess[:, 0], dim=0)
            pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
            hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        return mnode, hmess, agraph, bgraph 

    def forward(self, tree_tensors, inter_tensors, graph_tensors, mtree, minter, mgraph, subtree, subgraph):
        num_tree_nodes = tree_tensors[0].size(0)
        num_graph_nodes = graph_tensors[0].size(0)

        if len(subgraph[0]) + len(subgraph[1]) > 0:
            sub_graph_tensors = self.get_sub_tensor(graph_tensors, subgraph)[:-1]
            mgraph.node, mgraph.mess = self.graph_encoder(sub_graph_tensors, mgraph.mess, num_graph_nodes, subgraph)

        if len(subtree[0]) + len(subtree[1]) > 0:
            sub_inter_tensors = self.embed_sub_tree(inter_tensors, mgraph.node, subtree, is_inter_layer=True)
            minter.node, minter.mess = self.inter_encoder(sub_inter_tensors, minter.mess, num_tree_nodes, subtree)

            sub_tree_tensors = self.embed_sub_tree(tree_tensors, minter.node, subtree, is_inter_layer=False)
            mtree.node, mtree.mess = self.tree_encoder(sub_tree_tensors, mtree.mess, num_tree_nodes, subtree)

        return mtree, minter, mgraph

