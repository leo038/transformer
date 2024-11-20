import torch.nn as nn
import torch
import numpy as np

class GPTModel(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device):
        super(GPTModel, self).__init__()
        # 解码器
        self.decoder = Decoder(d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device)
        # 映射为词表大小
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, attention_mask=None):
        ##
        # inputs: [batch_size, seq_len]
        ##
        # outputs: [batch_size, seq_len, d_model]
        # self_attns: [n_layers, batch_size, n_heads, seq_len, seq_len]
        outputs, self_attns = self.decoder(inputs, attention_mask)
        # [batch_size, seq_len, vocab_size]
        logits = self.projection(outputs)
        return logits.view(-1, logits.size(-1)), self_attns


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device):
        super(Decoder, self).__init__()
        self.device = device
        # 将Token转为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_pos, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])

    def forward(self, inputs, attention_mask):
        ##
        # inputs: [batch_size, seq_len]
        ##
        # [batch_size, seq_len, d_model]
        outputs = self.embedding(inputs) + self.pos_encoding(inputs)
        # 上三角掩码，防止看到未来的信息， [batch_size, seq_len, seq_len]
        subsequence_mask = get_attn_subsequence_mask(inputs, self.device)
        if attention_mask is not None:
            # pad掩码 [batch_size, seq_len, seq_len]
            attention_mask = get_attn_pad_mask(attention_mask)
            # [batch_size, seq_len, seq_len]
            attention_mask = torch.gt((attention_mask + subsequence_mask), 0)
        else:
            attention_mask = subsequence_mask.bool()
        # 计算每一层的结果
        self_attns = []
        for layer in self.layers:
            # outputs: [batch_size, seq_len, d_model],
            # self_attn: [batch_size, n_heads, seq_len, seq_len],
            outputs, self_attn = layer(outputs, attention_mask)
            self_attns.append(self_attn)
        return outputs, self_attns


def get_attn_subsequence_mask(seq, device):
    # 注意力分数的大小是 [batch_size, n_heads, len_seq, len_seq]
    # 所以这里要生成 [batch_size, len_seq, len_seq] 大小
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 生成一个上三角矩阵
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask

def get_attn_pad_mask(attention_mask):
    batch_size, len_seq = attention_mask.size()
    attention_mask = attention_mask.data.eq(0).unsqueeze(1)
    # 注意力分数的大小是 [batch_size, n_heads, len_q, len_q]
    # 所以这里要转换成 [batch_size, len_seq, len_seq] 大小
    return attention_mask.expand(batch_size, len_seq, len_seq)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_pos, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(max_pos, d_model)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        # [seq_len] -> [batch_size, seq_len]
        pos = pos.unsqueeze(0).expand_as(inputs)
        return self.pos_embedding(pos)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v):
        super(DecoderLayer, self).__init__()
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        # 前馈神经网络层
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, inputs, attention_mask):
        ##
        # inputs: [batch_size, seq_len, d_model]
        # attention_mask: [batch_size, seq_len, seq_len]
        ##
        # outputs: [batch_size, seq_len, d_model]
        # self_attn: [batch_size, n_heads, seq_len, seq_len]
        outputs, self_attn = self.attention(inputs, inputs, inputs, attention_mask)
        # [batch_size, seq_len, d_model]
        outputs = self.pos_ffn(outputs)
        return outputs, self_attn
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        ##
        # inputs: [batch_size, seq_len, d_model]
        ##
        residual = inputs
        output = self.fc(inputs)
        # # LN + 残差计算, [batch_size, seq_len, d_model]
        return self.layernorm(output + residual)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attention_mask):
        ##
        # q: [batch_size, seq_len, d_model]
        # k: [batch_size, seq_len, d_model]
        # v: [batch_size, seq_len, d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        ##
        # 记录原始值, 后续计算残差
        residual, batch_size = q, q.size(0)
        # 先映射 q、k、v, 然后后分头
        # q: [batch_size, n_heads, len_q, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k: [batch_size, n_heads, len_k, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v: [batch_size, n_heads, len_v(=len_k), d_v]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # 点积注意力分数计算，  [batch_size, n_heads, len_q, d_v]
        context, attn = ScaledDotProductAttention(self.d_k)(q, k, v, attention_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 还原为原始大小
        output = self.fc(context)
        # LN + 残差计算
        return self.layernorm(output + residual), attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attention_mask):
        ##
        # q: [batch_size, n_heads, len_q, d_k]
        # k: [batch_size, n_heads, len_k, d_k]
        # v: [batch_size, n_heads, len_v, d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        ##
        # 计算每个Q与K的分数，计算出来的大小是 [batch_size, n_heads, len_q, len_q]
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 把被mask的地方置为无限小，softmax之后基本就是0，也就对q不起作用
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # 注意力后的大小 [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, v)
        return context, attn


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型参数
    model_param = {
        "d_model": 768,  # 嵌入层大小
        "d_ff": 2048,  # 前馈神经网络大小
        "d_k": 64,  # K 的大小
        "d_v": 64,  # V 的大小
        "n_layers": 6,  # 解码层的数量
        "n_heads": 8,  # 多头注意力的头数
        "max_pos": 1800,  # 位置编码的长度
        "device": device,  # 设备
        "vocab_size": 4825  # 词表大小
    }
    model = GPTModel(**model_param).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"模型参数量: {total_params/1e6} M")

    input_seq = torch.tensor([
        [2, 1, 3, 4, 7, 110],
        [3, 1, 3, 4, 9, 99]
    ]).to(device)

    res = model(input_seq)


if __name__ == '__main__':
    main()
