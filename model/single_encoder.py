import numpy as np

import torch
import torch.nn as nn

from torch_scatter import scatter_mean

import time

class Singularized_Relation_Encoder(nn.Module):
    def __init__(self, arities, dim_rel):
        super(Singularized_Relation_Encoder, self).__init__()

        self.embedding = nn.ParameterDict()
        for i_r in range(len(arities)):
            arity = arities[i_r]
            for i_emb in range(arity):
                rel = nn.Parameter(torch.FloatTensor(dim_rel))
                nn.init.normal_(rel, std=1/dim_rel**0.5)

                self.embedding[str((i_r, i_emb))] = rel

    def forward(self, batch_rels):
        u"""
        Args:
            batch_rels (list) -- A list of sigularized relations. Each relation is
                a tuple, (relation index, arg position).
        """
        out = []
        for rel in batch_rels:
            out.append(self.embedding[str(rel)].unsqueeze(0))
        out = torch.cat(out, dim=0)

        return out

class WrapperLSTM(nn.Module):
    def __init__(self, **args):
        super(WrapperLSTM, self).__init__()

        self.lstm = nn.LSTM(**args)

    def forward(self, h, torch_seq_lengths, max_length):
        n_seqs = h.size(0)
        h = torch.transpose(h, 0, 1) # (seq_len x batch_size x emb_dim)

        h = torch.nn.utils.rnn.pack_padded_sequence(h, torch_seq_lengths, batch_first=False)

        self.lstm.flatten_parameters()
        h, _ = self.lstm(h)

        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=False, total_length=max_length)

        h = h.view(max_length, n_seqs, 2, -1)
        h = torch.transpose(h, 0, 1) # (batch_size x seq_len x 2 x dim_lstm)
        h = h[:,:,0,:] + h[:,:,1,:] # (batch_size x seq_len x dim_lstm)

        return h

class Single_Text_Encoder(nn.Module):
    def __init__(self, n_token, dim_emb, dim_lstm, gpus):
        super(Single_Text_Encoder, self).__init__()

        self.dim_emb = dim_emb
        self.dim_lstm = dim_lstm

        # Look-up table of token embeddings.
        self.embedding = nn.Embedding(n_token, dim_emb, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=1/dim_emb**0.5)

        # LSTM
        # self.lstm = nn.LSTM(
        #     input_size = dim_emb,
        #     hidden_size = dim_lstm,
        #     num_layers = 1,
        #     bidirectional = True
        # )
        lstm = WrapperLSTM(
            input_size = dim_emb,
            hidden_size = dim_lstm,
            num_layers = 1,
            bidirectional = True
        )
        if gpus is not None:
            lstm = nn.DataParallel(lstm, device_ids=gpus)
        self.lstm = lstm

    def forward(self, batch_seqs, batch_positions):
        u"""
        Args:
            batch_seqs (list) -- A list of sequences. Each sequence is a tuple
                of token indices.
            batch_positions (list) -- A list of entity positions. Each entity
                position is represented as a tuple of position indices.
        """
        device = next(self.parameters()).device

        seq_lengths = [len(seq) for seq in batch_seqs]
        max_length = max(seq_lengths)
        seq_order = np.argsort(seq_lengths)[::-1] # Sequence indices. From longer to shorter.

        ## Padding sequences.
        padded_seqs = [
            list(batch_seqs[i_seq]) + [0] * (max_length - len(batch_seqs[i_seq]))
            for i_seq in seq_order
        ]
        padded_seqs = torch.LongTensor(padded_seqs).to(device)
        torch_seq_lengths = torch.LongTensor(sorted(seq_lengths, reverse=True)).to(device)

        ## Encode tokens.
        h = self.embedding(padded_seqs) # (batch_size x seq_len x emb_dim)

        ## Apply Bi-LSTM
        # h = torch.transpose(h, 0, 1) # (seq_len x batch_size x emb_dim)
        # h = torch.nn.utils.rnn.pack_padded_sequence(h, torch_seq_lengths, batch_first=False)
        #
        # h, _ = self.lstm(h)
        #
        # h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=False)
        # h = h.view(max_length, len(batch_seqs), 2, self.dim_lstm)
        # h = torch.transpose(h, 0, 1) # (batch_size x seq_len x 2 x dim_lstm)
        # h = h[:,:,0,:] + h[:,:,1,:] # (batch_size x seq_len x dim_lstm)
        h = self.lstm(h, torch_seq_lengths, max_length)

        # Reordering sequences to original order.
        # Mean representations at entity position.
        lst_now = []
        lst_to = []
        lst_pos = []
        for i_now, i_orig in enumerate(seq_order):
            for pos in batch_positions[i_orig]:
                lst_now.append(i_now)
                lst_to.append(i_orig)
                lst_pos.append(pos)
        lst_now = torch.LongTensor(lst_now).to(device)
        lst_to = torch.LongTensor(lst_to).to(device)
        lst_pos = torch.LongTensor(lst_pos).to(device)
        h = h[lst_now, lst_pos]

        # scatter_mean supports minus values.
        out = scatter_mean(h, lst_to, dim=0, dim_size=len(batch_seqs))

        return out
