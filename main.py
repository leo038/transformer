from torch import nn
import torch
import numpy as np

vocabulary_size = 200


def position_encoding(d_model=512, seq_len=2048):
    """
    This matches the implementation in the paper <Attention is All you Need>
    :param d_model: dimension of model size, in the paper, the model size equals to embedding size
    :param seq_len: the length of sequence
    :return: tensor with shape(seq_len, d_model)
    """
    x = np.linspace(0, d_model - 1, d_model)
    y = np.linspace(0, seq_len - 1, seq_len)
    X, Y = np.meshgrid(x, y)
    Z_even = np.sin(Y / (np.power(10000, X / d_model)))
    Z_odd = np.cos(Y / (np.power(10000, X / d_model)))

    for i in range(d_model // 2):
        Z_odd[:, 2 * i] = Z_even[:, 2 * i]
    return Z_odd


class PreProcess(nn.Module):

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=self.embedding_dim)

    def forward(self, input_seq):
        batch, seq_len = input_seq.shape
        embed = self.embedding(input_seq)  # batch, seq
        pos_encode = position_encoding(d_model=self.embedding_dim, seq_len=seq_len)
        res = embed + torch.tensor(pos_encode)
        return res


class MutiHeadAttention(nn.Module):

    def __init__(self, d_model=512, d_k=64, d_v=128):
        super().__init__()
        self.num_head = d_model // d_v
        self.d_k = d_k
        self.d_v = d_v
        if d_k == d_v:
            self.project_qkv = nn.Linear(in_features=d_model, out_features=3 * d_model, bias=False)
        else:
            ##if d_k!= d_v, we need to calculate q,k,v separately
            self.project_q = nn.Linear(in_features=d_model, out_features=self.d_k * self.num_head, bias=False)
            self.project_k = nn.Linear(in_features=d_model, out_features=self.d_k * self.num_head, bias=False)
            self.project_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.softmax = nn.Softmax()
        self.project_out = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(self, input_x):

        attention_out = None
        if self.d_k == self.d_v:
            qkv = self.project_qkv(input_x)  # the size of input_x is (batch, seq_len, d_model)
        else:
            q = self.project_q(input_x)
            k = self.project_k(input_x)
            v = self.project_v(input_x)
        for n_h in range(self.num_head):
            if self.d_k == self.d_v:
                Q = qkv[:, :, n_h * self.d_k: (n_h + 1) * self.d_k]
                K = qkv[:, :, (n_h + 1) * self.d_k: (n_h + 2) * self.d_k]
                V = qkv[:, :, (n_h + 2) * self.d_k: (n_h + 3) * self.d_k]
            else:
                Q = q[:, :, n_h * self.d_k: (n_h + 1) * self.d_k]
                K = k[:, :, n_h * self.d_k: (n_h + 1) * self.d_k]
                V = v[:, :, n_h * self.d_v: (n_h + 1) * self.d_v]
            attention_score = torch.bmm(Q, K.transpose(1, 2))
            attention_score = self.softmax(attention_score / np.sqrt(self.d_k))

            out = torch.bmm(attention_score, V)
            if attention_out is None:
                attention_out = out
            else:
                attention_out = torch.cat((attention_out, out), 2)  # 把多头结果进行concat
        out = self.project_out(attention_out)

        return out


class CrossMutiHeadAttention(nn.Module):

    def __init__(self, d_model=512, d_k=64, d_v=128):
        super().__init__()
        self.num_head = d_model // d_v
        self.d_k = d_k
        self.d_v = d_v

        self.project_q = nn.Linear(in_features=d_model, out_features=self.d_k * self.num_head, bias=False)
        self.project_k = nn.Linear(in_features=d_model, out_features=self.d_k * self.num_head, bias=False)
        self.project_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.softmax = nn.Softmax()
        self.project_out = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(self, encoder_output=None, pre_output=None):

        attention_out = None

        q = self.project_q(pre_output)
        k = self.project_k(encoder_output)
        v = self.project_v(encoder_output)
        for n_h in range(self.num_head):
            Q = q[:, :, n_h * self.d_k: (n_h + 1) * self.d_k]
            K = k[:, :, n_h * self.d_k: (n_h + 1) * self.d_k]
            V = v[:, :, n_h * self.d_v: (n_h + 1) * self.d_v]
            attention_score = torch.bmm(Q, K.transpose(1, 2))
            attention_score = self.softmax(attention_score / np.sqrt(self.d_k))

            out = torch.bmm(attention_score, V)
            if attention_out is None:
                attention_out = out
            else:
                attention_out = torch.cat((attention_out, out), 2)  # 把多头结果进行concat
        out = self.project_out(attention_out)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.mha = MutiHeadAttention()
        self.ffn = FeedForward()

    def forward(self, x):
        out = self.mha(x)
        out = self.layer_norm(out)
        out_mha = x + out

        out = self.ffn(out_mha)
        out = self.layer_norm(out)
        out = out_mha + out

        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.mha = MutiHeadAttention()
        self.cross_mha = CrossMutiHeadAttention()
        self.ffn = FeedForward()
        self.layer_nrom = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, encoder_output, pre_output):
        out = self.mha(pre_output)
        out = self.layer_nrom(out)
        out_mha = pre_output + out

        out = self.cross_mha(encoder_output=encoder_output, pre_output=out_mha)
        out = self.layer_nrom(out)
        out_cross_mha = out_mha + out

        out = self.ffn(out_cross_mha)
        out = self.layer_nrom(out)
        out = out_cross_mha + out
        return out


class Encoder(nn.Module):
    def __init__(self, encoder_layer_num=6):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(encoder_layer_num)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_layer_num=6):
        super().__init__()
        self.decoder = nn.ModuleList([DecoderLayer() for _ in range(decoder_layer_num)])

    def forward(self, encoder_output=None, pre_output=None):
        for layer in self.decoder:
            x = layer(encoder_output=encoder_output, pre_output=pre_output)
            pre_output = x
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.pre_process = PreProcess()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.out_project = nn.Linear(in_features=d_model, out_features=vocabulary_size)
        self.softmax = nn.Softmax()

    def forward(self, input_seq=None, pre_output=None):
        out = self.pre_process(input_seq).float()
        pre_output = self.pre_process(pre_output).float()
        encoder_output = self.encoder(out)
        out = self.decoder(encoder_output=encoder_output, pre_output=pre_output)
        out = self.out_project(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    input_seq = torch.tensor([
        [2, 1, 3, 4, 7, 110],
        [3, 1, 3, 4, 9, 99]
    ])

    pre_output = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ])

    transformer = Transformer()
    print("########################网络结构########################")
    print(transformer)
    print("######################################################")
    res = transformer(input_seq, pre_output)
    print(f"输出大小：{res.shape}")
