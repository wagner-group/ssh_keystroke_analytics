import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch.nn import (
    Parameter,
    TransformerEncoder,
    TransformerEncoderLayer,
)

# from train import *
torch.set_num_threads(1)


class ClassificationTransformer(nn.Module):
    def __init__(
        self,
        n_users,
        d_in,
        n_head,
        n_layers,
        dropout=0.1,
        normalizer="tan",
        partition=False,
        partition_count=4,
        partition_size=8,
        d_pos=-1,
        use_length=True,
        full=False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.partition_count = partition_count

        # Setup normalizer
        if type(normalizer) is str:
            self.norm = nn.ModuleList(
                [setup_normalizer(normalizer) for _ in range(d_in)]
            )
        else:
            self.norm = []
            for dim in normalizer:
                self.norm.append(setup_normalizer(dim[0], dim[1]))
            self.norm = nn.ModuleList(self.norm)

        # Figure out dimensions, set up partitions
        self.d_in = d_in
        self.d_pos = d_pos
        self.d_part = 0
        if partition:
            self.bins = nn.ModuleList(
                [
                    LearnablePartition(self.norm[i], bin_count=partition_size)
                    for i in range(self.d_in)
                    for _ in range(self.partition_count)
                ]
            )
            self.d_part = self.partition_count * self.d_in * partition_size
            self.pc = partition_count

        if full:
            self.d_ip = 64
        else:
            self.d_ip = 0

        if d_pos == -1:
            self.d_pos = (
                1 << (self.d_in + self.d_ip + self.d_part - 1).bit_length()
            ) - (self.d_in + self.d_part + self.d_ip)
            if self.d_pos % 2:
                self.d_pos += 1
            print(f"Set pos dim to {self.d_pos}")

        self.d_model = self.d_in + self.d_pos + self.d_part + self.d_ip
        self.n_users = n_users

        # Transformer Layer
        self.pos_encoder = PositionalEncoding(
            self.d_pos, self.d_model - self.d_pos, dropout
        )
        encoder_layers = TransformerEncoderLayer(
            self.d_model, n_head, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # FC layer
        self.fc1 = nn.Linear(self.d_model, n_users)

        # Classification token
        self.class_embedding = nn.Parameter(
            torch.rand(self.d_model - self.d_ip)
        )
        self.class_embedding.requiresGrad = True

        # Softmax and relu
        self.init_weights()

    def init_weights(self):
        initrange = 1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def forward(self, session, mask, offsets, src_ip=None, dst_ip=None):
        # input dimension : N x K x S x E
        # First item in sequence if empty to allow for encoding

        N, K, S, E = session.shape
        session = session.reshape(N * K, S, E)
        mask = mask.reshape(N * K, S)
        offsets = offsets.reshape(N * K)

        # Normalize each dimension, run through partition if necessary
        for dim in range(self.d_in):
            session[:, :, dim] = self.norm[dim](session[:, :, dim])

        if self.d_part > 0:
            for dim in range(self.d_in):
                session = torch.cat(
                    (
                        session,
                        torch.cat(
                            tuple(
                                self.bins[i](session[:, :, dim].unsqueeze(2))
                                for i in range(
                                    dim * self.pc, (dim + 1) * self.pc
                                )
                            ),
                            dim=2,
                        ),
                    ),
                    dim=2,
                )

        # Add position encoding and user encoding dimensions
        session = torch.cat(
            (
                session,
                torch.zeros((N * K, S, self.d_ip + self.d_pos)).to(
                    session.device
                ),
            ),
            dim=2,
        )

        # Add class embedding
        class_embed = self.class_embedding.unsqueeze(0).expand(
            N * K, self.d_model - self.d_ip
        )
        if self.d_ip:
            session[:, 0, : -self.d_ip] += class_embed
        else:
            session[:, 0] += class_embed

        # Add IP embedding
        if self.d_ip:
            if src_ip is not None:
                session[:, 0, -self.d_ip : -self.d_ip // 2] += src_ip
            if dst_ip is not None:
                session[:, 0, -self.d_in // 2 :] += dst_ip

        # Add positional encoding
        session = session.transpose(0, 1)
        output = self.pos_encoder(session, offsets)
        output = self.transformer_encoder(output, src_key_padding_mask=mask)[0]
        output = self.fc1(output.reshape(output.size(0), self.d_model))
        if K > 1:
            output = output.reshape(N, K, -1)
            output = torch.sum(output, dim=1)
        return output


class AuthenticationTransformer(nn.Module):
    def __init__(
        self,
        n_users,
        d_in,
        n_head,
        n_layers,
        dropout=0.1,
        normalizer="tan",
        partition=False,
        partition_count=4,
        partition_size=8,
        d_pos=-1,
        d_user=16,
        use_length=True,
        full=False,
        embed_lengths=False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.partition_count = partition_count
        self.el = embed_lengths

        # Length embedding
        if self.el:
            self.length_embedding = nn.Linear(
                1500, self.partition_count * partition_size
            )

        # Setup normalizer
        if type(normalizer) is str:
            self.norm = nn.ModuleList(
                [setup_normalizer(normalizer) for _ in range(d_in)]
            )
        else:
            self.norm = []
            for dim in normalizer:
                self.norm.append(setup_normalizer(dim[0], dim[1]))
            self.norm = nn.ModuleList(self.norm)

        # Figure out dimensions, set up partitions
        self.d_in = d_in
        self.d_pos = d_pos
        self.d_user = d_user
        self.d_part = 0
        if partition:
            self.bins = nn.ModuleList(
                [
                    LearnablePartition(self.norm[i], bin_count=partition_size)
                    for i in range(self.d_in)
                    for _ in range(self.partition_count)
                ]
            )
            self.d_part = self.partition_count * self.d_in * partition_size
            self.pc = partition_count

        if full:
            self.d_ip = 64
        else:
            self.d_ip = 0

        if d_pos == -1:
            self.d_pos = (
                1
                << (
                    self.d_in + self.d_ip + self.d_user + self.d_part - 1
                ).bit_length()
            ) - (self.d_in + self.d_part + self.d_user + self.d_ip)
            if self.d_pos % 2:
                self.d_pos -= 1
                self.d_user += 1
            print(f"Set pos dim to {self.d_pos}")

        self.d_model = (
            self.d_in + self.d_pos + self.d_part + self.d_user + self.d_ip
        )

        # Transformer Layer
        self.pos_encoder = PositionalEncoding(
            self.d_pos, self.d_model - self.d_pos, dropout
        )
        encoder_layers = TransformerEncoderLayer(
            self.d_model, n_head, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # FC layer
        self.fc1 = nn.Linear(self.d_model, 2)

        # User embedding
        self.n_users = n_users
        self.embedding = nn.Linear(n_users, self.d_user)

        # Softmax and relu
        self.init_weights()

        # Classification token
        self.class_embedding = nn.Parameter(
            torch.rand(self.d_model - (self.d_user + self.d_ip))
        )
        self.class_embedding.requiresGrad = True

        # Output Threshold
        self.thresholds = [0.5 for _ in range(n_users)]

    def init_weights(self):
        pass

    def prediction(self, output, users):
        # Take output form model, and use threhsolds to return prediction
        output = F.softmax(output, dim=1)[:, 0]
        users = torch.argmax(users, dim=1)
        predictions = []
        for i in range(output.size(0)):
            score = output[i].item()
            user = users[i].item()
            if not hasattr(self, "thresholds"):
                self.thresholds = [0.5 for _ in range(self.n_users)]
            if score > self.thresholds[user]:
                predictions.append(0.0)
            else:
                predictions.append(1.0)
        return torch.tensor(predictions).to(output.device)

    def forward(self, session, mask, offsets, users, src_ip=None, dst_ip=None):
        # input dimension : N x K x S x E
        # First item in sequence if empty to allow for encoding

        N, K, S, E = session.shape
        session = session.reshape(N * K, S, E)
        users = torch.repeat_interleave(users, K, dim=0)
        mask = mask.reshape(N * K, S)
        offsets = offsets.reshape(N * K)

        if self.el:
            L_in = torch.zeros(N * S, 1500).to(session.device).float()
            L_in[
                torch.arange(N * S), session.reshape(N * S, -1)[:, 0].long()
            ] = 1.0
            L_in = L_in.reshape(N, S, 1500)

        # Normalize each dimension, run through partition if necessary
        for dim in range(self.d_in):
            session[:, :, dim] = self.norm[dim](session[:, :, dim])

        if self.d_part > 0:
            partition_dims = [1] if self.el else range(self.d_in)
            for dim in partition_dims:
                session = torch.cat(
                    (
                        session,
                        torch.cat(
                            tuple(
                                self.bins[i](session[:, :, dim].unsqueeze(2))
                                for i in range(
                                    dim * self.pc, (dim + 1) * self.pc
                                )
                            ),
                            dim=2,
                        ),
                    ),
                    dim=2,
                )

        if self.el:
            session = torch.cat(
                (
                    session,
                    self.length_embedding(L_in.reshape(N * S, 1500)).reshape(
                        N, S, self.d_part // 2
                    ),
                ),
                dim=2,
            )

        # Add position encoding and user encoding dimensions
        session = torch.cat(
            (
                session,
                torch.zeros(
                    (N * K, S, self.d_user + self.d_ip + self.d_pos)
                ).to(session.device),
            ),
            dim=2,
        )

        # Add class embedding
        class_embed = self.class_embedding.unsqueeze(0).expand(
            N * K, self.d_model - (self.d_user + self.d_ip)
        )
        session[:, 0, : self.d_in + self.d_part] += class_embed[
            :, : self.d_in + self.d_part
        ]
        session[:, 0, -self.d_pos :] += class_embed[:, -self.d_pos :]

        # Add IP embedding
        if self.d_ip:
            if src_ip is not None:
                session[
                    :,
                    0,
                    self.d_in
                    + self.d_part : self.d_in
                    + self.d_part
                    + self.d_ip // 2,
                ] += src_ip
            if dst_ip is not None:
                session[
                    :,
                    0,
                    self.d_in
                    + self.d_part
                    + self.d_ip // 2 : self.d_in
                    + self.d_part
                    + self.d_ip,
                ] += dst_ip

        # Add user embedding
        session = session + torch.cat(
            (
                torch.zeros(N * K, S, self.d_in + self.d_part + self.d_ip).to(
                    session.device
                ),
                self.embedding(users)
                .unsqueeze(1)
                .expand(N * K, S, self.d_user),
                torch.zeros(N * K, S, self.d_pos).to(session.device),
            ),
            dim=2,
        )

        # Add positional encoding
        session = session.transpose(0, 1)
        output = self.pos_encoder(session, offsets)
        output = self.transformer_encoder(output, src_key_padding_mask=mask)[0]
        output = self.fc1(output.reshape(output.size(0), self.d_model))
        if K > 1:
            output = output.reshape(N, K, -1)
            output = torch.sum(output, dim=1) / K
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, d_ignore, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.cat((torch.zeros(max_len, 1, d_ignore), pe), dim=2)
        self.register_buffer("pe", pe)

    def forward(self, x, offsets):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        offsets = (offsets * 512).int()
        S, N, E = x.size(0), offsets.size(0), self.pe.size(2)
        pe = torch.zeros(S - 1, N, E).float().to(self.pe.device)
        for i in range(len(offsets)):
            pos_start = offsets[i]
            pos_end = min(pos_start + S - 1, self.pe.size(0))
            encoding_len = pos_end - pos_start
            pe[:encoding_len, i, :] = self.pe[pos_start:pos_end, 0, :]
        x[1:] = x[1:] + pe
        return self.dropout(x)


class LearnablePartition(nn.Module):
    def __init__(
        self,
        normalizer,
        bin_count=8,
        max_value=1.5,
        negative=False,
        offset_init=0,
    ):
        super().__init__()

        # f is the number of stddevs between bins
        self.f = Parameter(torch.rand(1) * 2.0 + 0.5)
        self.f.requiresGrad = True

        amplitude = max_value / (self.f * (bin_count))
        self.max = max_value
        self.negative = negative

        self.s = Parameter(torch.rand(bin_count) * amplitude)
        self.m = Parameter(torch.rand(1) * normalizer.f(offset_init))
        self.s.requiresGrad = True
        self.n = bin_count
        self.normalizer = normalizer
        self.m.requiresGrad = True

    def forward(self, x):
        N, S, _ = x.shape
        x = x.expand((N, S, self.n))
        s = torch.abs(self.s) + 0.00005
        if not self.negative:
            val = torch.cumsum(s * self.f, dim=0)[:-1] + self.m
            mean = (
                torch.cat((self.m, val), dim=0)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand((N, S, self.n))
            )
        else:
            val = (
                -torch.flip(
                    torch.cumsum(torch.flip(s * self.f, (0,)), dim=0)[:-1], (0,)
                )
                - self.mn
            )
            mean = (
                torch.cat((val, -self.m), dim=0)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand((N, S, self.n))
            )

        std = s.unsqueeze(0).unsqueeze(0).expand((N, S, self.n))
        pdf = torch.exp(-torch.square(x - mean) / (2 * torch.square(std)))
        # cdf = (torch.erf((x - mean) / (np.sqrt(2) * std)) + 1.0) * 0.5
        # return torch.cat((pdf,cdf*0.01), dim=2)
        return pdf

    def clip(self):
        # torch.clamp(self.s,0.001,2*self.max)
        # torch.clamp(self.f, 0.1, 100)
        pass

    def _to_pkt_size(self, x):
        return self.normalizer.fi(x).item()

    def output_size(self):
        return self.n

    def build_bins(self):
        s = torch.abs(self.s) + 0.00005
        if not self.negative:
            val = torch.cumsum(s * self.f, dim=0)[:-1] + self.m
            mean = torch.cat((self.m, val), dim=0)
        else:
            val = (
                -torch.flip(
                    torch.cumsum(torch.flip(s * self.f, (0,)), dim=0)[:-1], (0,)
                )
                - self.mn
            )
            mean = torch.cat((val, -self.m), dim=0)

        bins = [
            (mean[i] - s[i], mean[i], mean[i] + s[i])
            for i in range(mean.shape[0])
        ]
        bins = [
            (
                self._to_pkt_size(b[0]),
                self._to_pkt_size(b[1]),
                self._to_pkt_size(b[2]),
            )
            for b in bins
        ]
        return bins


class LogNormalizer(nn.Module):
    def __init__(self, m=100000):
        super().__init__()
        self.max = m
        # self.ratio.requiresGrad = False

    def forward(self, x):
        # return torch.log(x*self.ratio)*2/np.pi
        return self.f(x)

    def f(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0) / np.log(self.max)

    def fi(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.sign(x) * (
            torch.exp(torch.abs(x) * np.log(self.max)) - 1.0
        )


class TanNormalizer(nn.Module):
    def __init__(self, ratio=0.01):
        super().__init__()
        self.ratio = ratio
        self.max = np.pi / 2
        # self.ratio.requiresGrad = False

    def forward(self, x):
        return self.f(x)

    def f(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.arctan(x * self.ratio) / self.max

    def fi(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.tan(torch.clamp(x, -1.0, 1.0) * self.max) / self.ratio


class LinNormalizer(nn.Module):
    def __init__(self, ratio=1.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        return self.f(x)

    def f(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return x * self.ratio

    def fi(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return x / self.ratio


def ROC(model, data_points, targets, masks, positions, users):
    model.eval()  # turn on evaluation model
    results = None
    real = []
    with torch.no_grad():
        for batch in range(len(data_points)):
            tdata, target, mask, pos = (
                data_points[batch],
                targets[batch],
                masks[batch],
                positions[batch],
            )
            if results is None:
                results = F.softmax(
                    model(tdata, mask, pos, users[batch]), dim=1
                )[:, 1]
            else:
                results = torch.cat(
                    (
                        results,
                        F.softmax(model(tdata, mask, pos, users[batch]), dim=1)[
                            :, 1
                        ],
                    ),
                    dim=0,
                )
            real.extend(
                torch.argmax(target.squeeze(), dim=1).data.cpu().numpy()
            )

    results = results.data.cpu().numpy()
    real = np.array(real)
    return roc_curve(real, results)


def switch_device(device, *tensors):
    output = []
    for tensor in tensors:
        output.append(tensor.to(device))
    return tuple(output)


def purge_tensors(*tensors):
    for tensor in tensors:
        del tensor
    torch.cuda.empty_cache()


def save_conf_matrix(filename, m):
    with open(filename, "w") as outfile:
        outfile.write("\n".join("\t".join(str(a) for a in l) for l in m))


def setup_normalizer(name, param=None):
    normalizer = None
    if name == "tan":
        normalizer = TanNormalizer
    elif name == "log":
        normalizer = LogNormalizer
    else:
        normalizer = LinNormalizer
    return normalizer() if param is None else normalizer(param)


## Other work


## GRU from FS-Net
class FSNet(nn.Module):
    def __init__(self, n_users, d_model=128, dropout=0.1):
        super().__init__()
        self.model_type = "FSNet"
        self.d_model = d_model
        self.n_users = n_users

        # Embedding layers
        self.length_embedding = nn.Linear(1500, self.d_model)
        self.user_embedding = nn.Linear(self.n_users, self.d_model)
        self.time_embedding = nn.Linear(1, self.d_model)

        # Encoders
        self.lenc = nn.GRU(
            self.d_model, self.d_model, 2, dropout=dropout, bidirectional=True
        )
        self.tenc = nn.GRU(
            self.d_model, self.d_model, 2, dropout=dropout, bidirectional=True
        )

        # Decoders
        self.ldec = nn.GRU(
            4 * self.d_model,
            self.d_model,
            2,
            dropout=dropout,
            bidirectional=True,
        )
        self.tdec = nn.GRU(
            4 * self.d_model,
            self.d_model,
            2,
            dropout=dropout,
            bidirectional=True,
        )

        # Reconstruction
        # self.l_reconst = nn.Linear(2*self.d_model, 1500)
        # self.time_reconst = nn.Linear(2*self.d_model, 1)

        # Dense layer
        self.selu = nn.SELU()
        self.dl1 = nn.Linear(33 * self.d_model, 10 * self.d_model)
        self.dl2 = nn.Linear(10 * self.d_model, self.d_model)

        # Classification layer
        self.classif = nn.Linear(self.d_model, 2)

        # Rec loss
        # self.rloss = nn.CrossEntropyLoss()

        self.thresholds = [0.5 for _ in range(n_users)]

    def forward(self, x, users):
        # Expected format GRU : (L, N, H)
        # Input format : N x 1 x S x 2

        N, _, S, _ = x.shape
        x = x.reshape(N, S, -1)
        L_in = torch.zeros(N * S, 1500).to(x.device).float()
        L_in[torch.arange(N * S), x.reshape(N * S, -1)[:, 0].long()] = 1.0
        L_in = L_in.reshape(N, S, 1500).transpose(0, 1)
        T_in = x[:, :, 1].transpose(0, 1).reshape(S, N, 1)
        U_in = users

        # print(f"Lin : {L_in}\nT_in : {T_in}\nU_in : {U_in}")

        # Embedding
        L = self.length_embedding(L_in)
        T = self.time_embedding(T_in)
        U = self.user_embedding(U_in)

        # print(f"L : {L}\nT : {T}\nU : {U}")

        # Encoder
        _, LZE = self.lenc(L)
        LZE = LZE.transpose(0, 1).reshape(N, -1)

        _, TZE = self.tenc(T)
        TZE = TZE.transpose(0, 1).reshape(N, -1)

        # Decoder
        LD, LZD = self.ldec(LZE.reshape(1, N, -1).expand(S, N, -1))
        LZD = LZD.transpose(0, 1).reshape(N, -1)

        TD, TZD = self.ldec(TZE.reshape(1, N, -1).expand(S, N, -1))
        TZD = TZD.transpose(0, 1).reshape(N, -1)

        # Reconstuction
        # LR = self.l_reconst(LD).reshape(N*S,1500)
        # lloss = self.rloss(LR, x.reshape(N*S, -1)[:, 0].long()) / np.log(1500)

        # TR = self.time_reconst(TD).reshape(N*S,1)
        # tloss = torch.arctan(torch.square(TR - x[:, :, 1].reshape(N*S, 1)).mean()*10) * 2 / np.pi
        # reconstructed = (lloss + tloss) / 2

        # Dense Layer
        DL_in = torch.cat(
            (
                LZE,
                LZD,
                LZE * LZD,
                torch.abs(LZE - LZD),
                TZE,
                TZD,
                TZE * TZD,
                torch.abs(TZE - TZD),
                U,
            ),
            dim=1,
        )
        DL_out = F.selu(self.dl2(F.selu(self.dl1(DL_in))))

        # Prediction
        output = self.classif(DL_out)

        # print(f"Output : {output}")

        # return output, reconstructed
        return output, None

    def prediction(self, output, users):
        # Take output form model, and use threhsolds to return prediction
        output = F.softmax(output, dim=1)[:, 0]
        users = torch.argmax(users, dim=1)
        predictions = []
        for i in range(output.size(0)):
            score = output[i].item()
            user = users[i].item()
            if not hasattr(self, "thresholds"):
                self.thresholds = [0.5 for _ in range(self.n_users)]
            if score > self.thresholds[user]:
                predictions.append(0.0)
            else:
                predictions.append(1.0)
        return torch.tensor(predictions).to(output.device)


## CNN+GRU from Continuous authentication by free-text keystroke based on CNN and RNN
class CNNGRU(nn.Module):
    def __init__(self, n_users, d_model=256, dropout=0.1, cnn_ker=5):
        super().__init__()
        self.model_type = "CNNGRU"

        self.d_model = d_model
        self.n_users = n_users
        self.ker = cnn_ker

        # Embedding
        self.user_embedding = nn.Linear(self.n_users, self.d_model)

        # Normalization
        self.norm = nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=False)

        # Convolution
        self.conv = nn.Conv1d(2, d_model, self.ker)
        self.pool = nn.AvgPool1d(2)

        # GRU
        self.enc = nn.GRU(d_model, self.d_model, 2, dropout=dropout)

        # Classif
        self.classif = nn.Linear(2 * self.d_model, 2)
        self.thresholds = [0.5 for _ in range(n_users)]

    def forward(self, x, users):
        # Expected format GRU : (L, N, H)
        # Input format : N x 1 x S x 2

        N, _, S, _ = x.shape
        x = x.reshape(N, S, 2)
        x = x.transpose(1, 2)

        # Norm
        x[:, 0, :] = x[:, 0, :] / 1500
        x = self.norm(x)

        # Conv
        x = self.conv(x)

        # Pool
        x = self.pool(x)

        # Encoder
        x = x.transpose(1, 2).transpose(0, 1)
        x, _ = self.enc(x)
        x = x[-1, :, :].squeeze()

        # Classification
        output = self.classif(torch.cat((x, self.user_embedding(users)), dim=1))
        return output

    def prediction(self, output, users):
        # Take output form model, and use threhsolds to return prediction
        output = F.softmax(output, dim=1)[:, 0]
        users = torch.argmax(users, dim=1)
        predictions = []
        for i in range(output.size(0)):
            score = output[i].item()
            user = users[i].item()
            if not hasattr(self, "thresholds"):
                self.thresholds = [0.5 for _ in range(self.n_users)]
            if score > self.thresholds[user]:
                predictions.append(0.0)
            else:
                predictions.append(1.0)
        return torch.tensor(predictions).to(output.device)


## LSTM from TypeNet
class TypeNet(nn.Module):
    def __init__(self, n_users, d_model=128):
        super().__init__()
        self.model_type = "TypeNet"
        self.d_model = d_model
        self.n_users = n_users

        # Embedding layers
        self.user_embedding = nn.Linear(self.n_users, self.d_model - 2)

        # Batch norm
        self.bn1 = nn.BatchNorm1d(d_model)

        # Encoders
        self.enc = nn.LSTM(self.d_model, self.d_model, 2, dropout=0.2)

        # FCN
        self.fcn = nn.Linear(self.d_model, 2)

        self.thresholds = [0.5 for _ in range(n_users)]

    def forward(self, session, users, positions):
        N, K, S, E = session.shape
        session = session.reshape(N * K, S, E)
        users = torch.repeat_interleave(users, K, dim=0)

        session = torch.cat(
            (
                session,
                self.user_embedding(users)
                .unsqueeze(1)
                .expand(N * K, S, self.d_model - 2),
            ),
            dim=2,
        )
        session = self.bn1(session.permute(0, 2, 1)).permute(0, 2, 1)

        # Pack sequence
        session = nn.utils.rnn.pack_padded_sequence(
            session, positions, batch_first=True, enforce_sorted=False
        )

        # LSTM
        _, (output, _) = self.enc(session)

        # FCN
        output = self.fcn(output[0])

        if K > 1:
            output = output.reshape(N, K, -1)
            output = torch.sum(output, dim=1) / K
        return output

    def prediction(self, output, users):
        # Take output form model, and use threhsolds to return prediction
        output = F.softmax(output, dim=1)[:, 0]
        users = torch.argmax(users, dim=1)
        predictions = []
        for i in range(output.size(0)):
            score = output[i].item()
            user = users[i].item()
            if not hasattr(self, "thresholds"):
                self.thresholds = [0.5 for _ in range(self.n_users)]
            if score > self.thresholds[user]:
                predictions.append(0.0)
            else:
                predictions.append(1.0)
        return torch.tensor(predictions).to(output.device)
