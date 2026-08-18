import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFeatureAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x shape: (batch, hidden_dim)
        weights = self.attn(x)
        return x * weights

class DNNBase(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=128, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        if self.use_attention:
            self.attention = DynamicFeatureAttention(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        if self.use_attention:
            x = self.attention(x)
        x = self.dropout(x)
        return self.fc2(x)

class CNNBase(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=128, kernel_size=3, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        if self.use_attention:
            self.attention = DynamicFeatureAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        x, _ = torch.max(x, dim=2)
        if self.use_attention:
            x = self.attention(x)
        x = self.dropout(x)
        return self.fc(x)

class BiCNNBase(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=128, kernel_size=3, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.conv_fwd = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv_bwd = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        if self.use_attention:
            self.attention = DynamicFeatureAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x_transpose = x.transpose(1, 2)
        fwd = F.relu(self.conv_fwd(x_transpose))
        fwd, _ = torch.max(fwd, dim=2)
        
        x_rev = torch.flip(x_transpose, dims=[2])
        bwd = F.relu(self.conv_bwd(x_rev))
        bwd, _ = torch.max(bwd, dim=2)
        
        out = torch.cat([fwd, bwd], dim=1)
        if self.use_attention:
            out = self.attention(out)
        out = self.dropout(out)
        return self.fc(out)

class RNNBase(nn.Module):
    def __init__(self, rnn_type, embed_dim, num_classes, hidden_dim=128, bidirectional=False, use_attention=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
            
        fc_in = hidden_dim * 2 if bidirectional else hidden_dim
        if self.use_attention:
            self.attention = DynamicFeatureAttention(fc_in)
        self.fc = nn.Linear(fc_in, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out.mean(dim=1)
        if self.use_attention:
            out = self.attention(out)
        out = self.dropout(out)
        return self.fc(out)

def get_model(algo, embed_dim, num_classes, use_attention=False):
    if algo == 'DNN':
        return DNNBase(embed_dim, num_classes, use_attention=use_attention)
    elif algo == 'CNN':
        return CNNBase(embed_dim, num_classes, use_attention=use_attention)
    elif algo == 'BiCNN':
        return BiCNNBase(embed_dim, num_classes, use_attention=use_attention)
    elif algo == 'LSTM':
        return RNNBase('LSTM', embed_dim, num_classes, bidirectional=False, use_attention=use_attention)
    elif algo == 'BiLSTM':
        return RNNBase('LSTM', embed_dim, num_classes, bidirectional=True, use_attention=use_attention)
    elif algo == 'GRU':
        return RNNBase('GRU', embed_dim, num_classes, bidirectional=False, use_attention=use_attention)
    else:
        raise ValueError(f"Unknown algo {algo}")
