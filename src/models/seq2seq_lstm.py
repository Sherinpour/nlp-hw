import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, num_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)

    def forward(self, src):
        emb = self.emb(src)
        outputs, (h, c) = self.rnn(emb)
        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, num_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc  = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, h, c):
        emb = self.emb(x)  # (B,1,E)
        out, (h, c) = self.rnn(emb, (h, c))
        logits = self.fc(out)  # (B,1,V)
        return logits, h, c

class Seq2Seq(nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder, sos_idx: int, eos_idx: int, teacher_forcing: float=0.5):
        super().__init__()
        self.enc, self.dec = enc, dec
        self.sos, self.eos = sos_idx, eos_idx
        self.tf = teacher_forcing

    def forward(self, src, tgt):
        B, T = tgt.size()
        h, c = self.enc(src)
        inp = tgt[:, 0].unsqueeze(1)  # <sos>
        outputs = []
        for t in range(1, T):
            logits, h, c = self.dec(inp, h, c)
            outputs.append(logits)
            next_tok = logits.argmax(-1)
            use_tf = torch.rand(1).item() < self.tf
            inp = tgt[:, t].unsqueeze(1) if use_tf else next_tok
        return torch.cat(outputs, dim=1)  # (B, T-1, V)
