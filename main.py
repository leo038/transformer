from torch import nn
import torch
import numpy as np

vocabulary_size = 200


def position_encoding(d_model=512, seq_len=2048):
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

    def __init__(self, d_model=512, num_head=8, d_k=64, d_v=64):
        super().__init__()
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v
        if d_k == d_v:
            self.project_qkv = nn.Linear(in_features=d_model, out_features=3 * d_model, bias=False)
        self.softmax = nn.Softmax()
        self.project_out = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(self, input_x):
        qkv = self.project_qkv(input_x)  # the size of input_x is (batch, seq_len, d_model)

        attention_out = None
        for n_h in range(self.num_head):
            Q = qkv[:, :, n_h * self.d_k: (n_h + 1) * self.d_k]
            K = qkv[:, :, (n_h + 1) * self.d_k: (n_h + 2) * self.d_k]
            V = qkv[:, :, (n_h + 2) * self.d_k: (n_h + 3) * self.d_k]
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
    def __init__(self, d_model=512, d_ff=2048):
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


class Encoder(nn.Module):
    def __init__(self, encoder_layer_num=6):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(encoder_layer_num)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = PreProcess()
        self.encoder = Encoder()

    def forward(self, input_seq):
        out = self.pre_process(input_seq).float()
        out = self.encoder(out)
        return out


if __name__ == "__main__":
    input_seq = torch.tensor([
        [2, 1, 3, 4, 7, 110],
        [3, 1, 3, 4, 9, 99]
    ])

    transformer = Transformer()
    print("########################网络结构########################")
    print(transformer)
    print("######################################################")
    res = transformer(input_seq)
    print(f"输出大小：{res.shape}")
