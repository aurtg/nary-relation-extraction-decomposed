import numpy as np

import torch
import torch.nn as nn

from torch_scatter import scatter_max

class Binarized_Relation_Encoder(nn.Module):
    def __init__(self, arities, dim_rel):
        super(Binarized_Relation_Encoder, self).__init__()

        self.embedding = nn.ParameterDict()
        for i_r in range(len(arities)):
            arity = arities[i_r]
            for i_emb1 in range(arity):
                for i_emb2 in range(arity):
                    if i_emb1 == i_emb2:
                        continue

                    rel = nn.Parameter(torch.FloatTensor(dim_rel))
                    nn.init.normal_(rel, std=1/dim_rel**0.5)

                    self.embedding[str((i_r, i_emb1, i_emb2))] = rel

    def forward(self, batch_rels):
        u"""
        Args:
            batch_rels (list) -- A list of binarized relations. Each relation is
                a tuple, (relation index, arg1 position, arg2 position).
        """
        out = []
        for rel in batch_rels:
            out.append(self.embedding[str(rel)].unsqueeze(0))
        out = torch.cat(out, dim=0)

        return out

    def normalize(self):
        with torch.no_grad():
            for key in self.embedding:
                param = self.embedding[key]
                param.div_(torch.norm(param))


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

#NOTE: padding index must be zero.
class Path_Encoder(nn.Module):
    def __init__(self, n_token, dim_emb, dim_lstm, gpus):
        super(Path_Encoder, self).__init__()

        print(gpus)

        self.dim_emb = dim_emb
        self.dim_lstm = dim_lstm

        # Look-up table of token embeddings.
        self.embedding = nn.Embedding(n_token, dim_emb, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=1/dim_emb**0.5)

        # LSTM
        lstm = WrapperLSTM(
            input_size = dim_emb,
            hidden_size = dim_lstm,
            num_layers = 1,
            bidirectional = True
        )
        if gpus is not None:
            lstm = nn.DataParallel(lstm, device_ids=gpus)
        self.lstm = lstm

    def forward(self, batch_seqs):
        u"""
        Args:
            batch_seqs (list) -- A list of sequences. Each sequence is a tuple (or list)
                of token indices.
        """
        device = next(self.parameters()).device

        seq_lengths = [len(seq) for seq in batch_seqs]
        max_length = max(seq_lengths)
        seq_order = np.argsort(seq_lengths)[::-1] # Sequence indices. From longer one to shorter one.

        ## Padding sequences.
        padded_seqs = [
            list(batch_seqs[i_seq]) + [0] * (max_length - len(batch_seqs[i_seq]))
            for i_seq in seq_order
        ]
        padded_seqs = torch.LongTensor(padded_seqs).to(device)
        torch_seq_lengths = torch.LongTensor(sorted(seq_lengths, reverse=True)).to(device)

        ## Encode tokens.
        h = self.embedding(padded_seqs) # (batch_size x seq_len x emb_dim)

        h = self.lstm(h, torch_seq_lengths, max_length)

        # Reordering sequences to original order, and apply max pooling.
        lst_now = []
        lst_to = []
        lst_pos = []
        for i_now, i_orig in enumerate(seq_order):
            len_seq = seq_lengths[i_orig]

            lst_now += [i_now] * len_seq
            lst_to += [i_orig] * len_seq
            lst_pos += range(len_seq)
        lst_now = torch.LongTensor(lst_now).to(device)
        lst_to = torch.LongTensor(lst_to).to(device)
        lst_pos = torch.LongTensor(lst_pos).to(device)

        h = h[lst_now, lst_pos]

        min_h = torch.min(h).item()

        # print("="*40)
        # print(h)
        # print(lst_to)
        # scatter_max does not support minus values.
        out, _ = scatter_max(h-min_h, lst_to, dim=0, dim_size=len(batch_seqs))
        out = out + min_h
        #print(out)

        return out

class Text_Encoder_(nn.Module):
    def __init__(self, n_token, dim_emb, dim_lstm, dim_rel):
        super(Text_Encoder_, self).__init__()

        self.dim_emb = dim_emb
        self.dim_lstm = dim_lstm

        self.embedding = nn.Embedding(n_token, dim_emb, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=1/dim_emb**0.5)

        self.lstm = nn.LSTM(
            input_size = dim_emb,
            hidden_size = dim_lstm,
            num_layers = 1,
            bidirectional = True
        )

        self.aggregate = nn.Linear(2*dim_lstm, dim_rel)

    def forward(self, batch_seqs, batch_pos):
        u"""
        Args:
            batch_seqs (list) -- A list of sequences. Each sequence is a tuple (or list)
                of token indices.
            batch_pos (list) -- A list of position indices of entities in sequences.
                E.g. [((0,1,2), (5,6)), ...] # (0,1,2) is the position indices of
                the first entity in the first sequence, while (5,6) is that of the
                second entity in the sequence.
        """
        device = next(self.parameters()).device

        seq_lengths = [len(seq) for seq in batch_seqs]
        max_length = max(seq_lengths)
        seq_order = np.argsort(seq_lengths)[::-1]

        ## Padding sequences.
        padded_seqs = [
            list(batch_seqs[i_seq]) + [0] * (max_length - len(batch_seqs[i_seq]))
            for i_seq in seq_order
        ]
        padded_seqs = torch.LongTensor(padded_seqs).to(device)
        torch_seq_lengths = torch.LongTensor(sorted(seq_lengths, reverse=True)).to(device)

        ## Embedding.
        h = self.embedding(padded_seqs)
        h = torch.transpose(h, 0, 1)

        ## Apply Bi-LSTM
        h = torch.nn.utils.rnn.pack_padded_sequence(h, torch_seq_lengths, batch_first=False)
        h, _ = self.lstm(h)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=False)

        h = h.view(max_length, len(batch_seqs), 2, self.dim_lstm)
        h = torch.transpose(h, 0, 1) # (batch_size x seq_len x 2 x dim_lstm)
        h = h[:,:,0,:] + h[:,:,1,:] # (batch_size x seq_len x dim_lstm)

        ## Reordering and aggregation.
        out = [None]*len(batch_seqs)
        for i_now, i_orig in enumerate(seq_order):
            pos1, pos2 = batch_pos[i_orig]
            pos1, pos2 = torch.LongTensor(pos1).to(device), torch.LongTensor(pos2).to(device)

            rep_ent1 = torch.mean(h[i_now][pos1], dim=0)
            rep_ent1 = rep_ent1.view(1,-1)
            rep_ent2 = torch.mean(h[i_now][pos2], dim=0)
            rep_ent2 = rep_ent2.view(1,-1)

            # rep = self.aggregate(
            #     torch.cat([rep_ent1, rep_ent2], dim=1)
            # )
            rep, _ = torch.max(torch.cat([rep_ent1, rep_ent2], dim=0), keepdim=True)
            out[i_orig] = rep
        out = torch.cat(out, dim=0)

        return out


if __name__=="__main__":
    m = Path_Encoder(5, 5, 2)

    m([[1,2,3],[1,4,2,3],[2,3]])
