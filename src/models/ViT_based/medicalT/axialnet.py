import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import qkv_transform

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        super(AxialAttention, self).__init__()
        self.in_planes, self.out_planes, self.groups, self.kernel_size, self.stride = in_planes, out_planes, groups, kernel_size, stride
        self.group_planes = out_planes // groups
        self.width = width
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index, key_index = torch.arange(kernel_size).unsqueeze(0), torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1: self.pooling = nn.AvgPool2d(stride, stride=stride)
        self.reset_parameters()

    def forward(self, x):
        if self.width: x = x.permute(0, 2, 1, 3)
        else: x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)
        if self.width: output = output.permute(0, 2, 1, 3)
        else: output = output.permute(0, 2, 3, 1)
        if self.stride > 1: output = self.pooling(output)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialBlock(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv_down(x)))
        out = self.relu(self.width_block(self.hight_block(out)))
        out = self.bn2(self.conv_up(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResAxialAttentionUNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, s=0.125, img_size=224, imgchan=1):
        super(ResAxialAttentionUNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.inplanes = int(64 * s)
        self.dilation = 1
        self.groups = 8
        self.base_width = 64
        block_expansion = block.expansion

        # --- Encoder Part ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2))
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size // 4))
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size // 8))
        
        # --- Decoder Part ---
        self.decoder_conv4 = nn.Conv2d(int(1024 * s * block_expansion), int(512 * s * block_expansion), kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(int(512 * s * block_expansion), int(256 * s * block_expansion), kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(int(256 * s * block_expansion), int(128 * s * block_expansion), kernel_size=3, padding=1)
        self.decoder_conv1 = nn.Conv2d(int(128 * s * block_expansion), int(64 * s), kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(int(64 * s), num_classes, kernel_size=1)

        # Grouping for Freezing
        self.encoder = nn.Sequential(self.encoder_conv, self.layer1, self.layer2, self.layer3, self.layer4)
        self.decoder = nn.Sequential(self.decoder_conv4, self.decoder_conv3, self.decoder_conv2, self.decoder_conv1, self.final_conv)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, kernel_size=kernel_size)]
        self.inplanes = planes * block.expansion
        if stride != 1: kernel_size //= 2
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x_conv = self.encoder_conv(x)
        x1 = self.layer1(x_conv)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder
        d4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = self.decoder_conv4(d4)
        d4 = torch.add(d4, x3)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.decoder_conv3(d3)
        d3 = torch.add(d3, x2)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.decoder_conv2(d2)
        d2 = torch.add(d2, x1)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.decoder_conv1(d1)
        
        # --- FIX: Remove the extra upsampling step ---
        # The output of d1 is now 224x224 after the final convolution.
        # It should NOT be upsampled again.
        out = self.final_conv(d1)
        return out

def MedT(**kwargs):
    model = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], **kwargs)
    return model