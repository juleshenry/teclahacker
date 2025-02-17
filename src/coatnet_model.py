import torch
import torch.nn as nn
from torch.nn import functional as F

class MBConv(nn.Module):
    """MBConv block with expansion ratio of 4"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        expanded = in_channels * 4
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.GELU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection if input and output dimensions match
        self.skip = None
        if stride == 1 and in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.skip is not None:
            x = self.skip(x)
        if x.shape == out.shape:
            out = out + x
        return out

class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position encoding"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 32, 32))
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with relative position bias
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias[:, :N, :N]
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with relative position encoding"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelativeMultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CoAtNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, dim=256):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim//2),
            nn.GELU()
        )
        
        # First stage: MBConv blocks
        self.stage1 = nn.Sequential(
            MBConv(dim//2, dim//2),
            MBConv(dim//2, dim)
        )
        
        # Second stage: MBConv blocks
        self.stage2 = nn.Sequential(
            MBConv(dim, dim, stride=2),
            MBConv(dim, dim)
        )
        
        # Convert to sequence for transformer
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        
        # Third stage: Transformer blocks
        self.stage3 = nn.Sequential(
            TransformerBlock(dim),
            TransformerBlock(dim)
        )
        
        # Fourth stage: Transformer blocks
        self.stage4 = nn.Sequential(
            TransformerBlock(dim),
            TransformerBlock(dim)
        )
        
        # Final classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        # Convolutional stages
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        
        # Convert to sequence
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        x = self.to_patch_embedding(x)
        
        # Transformer stages
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Classification
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

# Example usage
if __name__ == "__main__":
    # Create a model instance
    model = CoAtNet(num_classes=36)  # 36 classes for keys 0-9 and a-z
    
    # Test with a sample input
    batch_size = 16
    input_channels = 1  # For mel-spectrograms
    height = 64
    width = 64
    x = torch.randn(batch_size, input_channels, height, width)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
