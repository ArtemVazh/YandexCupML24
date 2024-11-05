from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer, Emformer
import math
from torchvision.models import resnet50
from transformers import ASTConfig, ASTModel
from transformers import EfficientNetModel
import timm

import librosa
import numpy as np

from typing import Tuple, Dict, Optional

class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self, in_channels: int, out_channels: int, last: bool = False, downsample=None, stride=1, bias: bool = True
    ):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        if not last:
            # Apply Instance normalization in first half channels (ratio=0.5)
            self.ibn = IBN(out_channels, ratio=0.5)
        else:
            self.ibn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x.clone()

        x = self.conv1(x)
        x = self.ibn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = residual + x
        out = self.relu(out)

        return out


class Resnet50(nn.Module):
    def __init__(
        self,
        ResBlock: Bottleneck,
        emb_dim: int = 1024,
        num_channels: int = 1,
        dropout=0.1,
        n_bins=84
    ) -> None:

        super(Resnet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, blocks=3, planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, blocks=4, planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, blocks=6, planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, blocks=3, planes=256, stride=1, last=True)

    def _make_layer(self, ResBlock: Bottleneck, blocks: int, planes: int, stride: int = 1, last: bool = False):
        downsample = None
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )
        layers = []
        layers.append(
            ResBlock(in_channels=self.in_channels, out_channels=planes, stride=stride, downsample=downsample, last=last)
        )
        self.in_channels = planes * ResBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResBlock(in_channels=self.in_channels, out_channels=planes, last=last))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # Unsqueeze to simulate 1-channel image
        x = self.conv1(x.unsqueeze(1))
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class HardTripletLoss(nn.Module):
  """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

  def __init__(self, margin=0.1):
    """ Args:
      margin: margin for triplet loss
    """
    super(HardTripletLoss, self).__init__()
    self._margin = margin
    return

  def forward(self, embeddings, labels):
    """
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = self._pairwise_distance(embeddings, squared=False)

    mask_anchor_positive = self._get_anchor_positive_triplet_mask(
      labels).float()
    valid_positive_dist = pairwise_dist * mask_anchor_positive
    hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1,
                                         keepdim=True)

    # Get the hardest negative pairs
    mask_negative = self._get_anchor_negative_triplet_mask(labels).float()
    max_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    negative_dist = pairwise_dist + max_negative_dist * (1.0 - mask_negative)
    hardest_negative_dist, _ = torch.min(negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = F.relu(
      hardest_positive_dist - hardest_negative_dist + self._margin)
    triplet_loss = torch.mean(triplet_loss)
    return triplet_loss

  @staticmethod
  def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
      mask = torch.eq(distances, 0.0).float()
      distances = distances + mask * eps
      distances = torch.sqrt(distances)
      distances = distances * (1.0 - mask)
    return distances

  @staticmethod
  def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True, if a and p are distinct and
       have same label.

    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = labels.device
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = indices_not_equal * labels_equal
    return mask

  @staticmethod
  def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    """
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1
    return mask

class AttentiveStatisticsPooling(torch.nn.Module):
  """This class implements an attentive statistic pooling layer for each channel.
  It returns the concatenated mean and std of the input tensor.

  Arguments
  ---------
  channels: int
      The number of input channels.
  output_channels: int
      The number of output channels.
  """

  def __init__(self, channels, output_channels):
    super().__init__()

    self._eps = 1e-12
    self._linear = torch.nn.Linear(channels * 3, channels)
    self._tanh = torch.nn.Tanh()
    self._conv = torch.nn.Conv1d(
      in_channels=channels, out_channels=channels, kernel_size=1
    )
    self._final_layer = torch.nn.Linear(channels * 2, output_channels,
                                        bias=False)
    return

  @staticmethod
  def _compute_statistics(x: torch.Tensor,
                          m: torch.Tensor,
                          eps: float,
                          dim: int = 2):
    mean = (m * x).sum(dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std

  def forward(self, x: torch.Tensor):
    """Calculates mean and std for a batch (input tensor).

    Args:
      x : torch.Tensor
          Tensor of shape [N, L, C].
    """

    x = x.transpose(1, 2)
    L = x.shape[-1]
    lengths = torch.ones(x.shape[0], device=x.device)
    mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
    mask = mask.unsqueeze(1)
    total = mask.sum(dim=2, keepdim=True).float()

    mean, std = self._compute_statistics(x, mask / total, self._eps)
    mean = mean.unsqueeze(2).repeat(1, 1, L)
    std = std.unsqueeze(2).repeat(1, 1, L)
    attn = torch.cat([x, mean, std], dim=1)
    attn = self._conv(self._tanh(self._linear(
      attn.transpose(1, 2)).transpose(1, 2)))

    attn = attn.masked_fill(mask == 0, float("-inf"))  # Filter out zero-padding
    attn = F.softmax(attn, dim=2)
    mean, std = self._compute_statistics(x, attn, self._eps)
    pooled_stats = self._final_layer(torch.cat((mean, std), dim=1))
    return pooled_stats

  def forward_with_mask(self, x: torch.Tensor,
                        lengths: Optional[torch.Tensor] = None):
    """Calculates mean and std for a batch (input tensor).

    Args:
      x : torch.Tensor
          Tensor of shape [N, C, L].
      lengths:
    """
    L = x.shape[-1]

    if lengths is None:
      lengths = torch.ones(x.shape[0], device=x.device)

    # Make binary mask of shape [N, 1, L]
    mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
    mask = mask.unsqueeze(1)

    # Expand the temporal context of the pooling layer by allowing the
    # self-attention to look at global properties of the utterance.

    # torch.std is unstable for backward computation
    # https://github.com/pytorch/pytorch/issues/4320
    total = mask.sum(dim=2, keepdim=True).float()
    mean, std = self._compute_statistics(x, mask / total, self._eps)

    mean = mean.unsqueeze(2).repeat(1, 1, L)
    std = std.unsqueeze(2).repeat(1, 1, L)
    attn = torch.cat([x, mean, std], dim=1)

    # Apply layers
    attn = self.conv(self._tanh(self._linear(attn, lengths)))

    # Filter out zero-paddings
    attn = attn.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(attn, dim=2)
    mean, std = self._compute_statistics(x, attn, self._eps)
    # Append mean and std of the batch
    pooled_stats = torch.cat((mean, std), dim=1)
    pooled_stats = pooled_stats.unsqueeze(2)
    return pooled_stats

  @staticmethod
  def length_to_mask(length: torch.Tensor,
                     max_len: Optional[int] = None,
                     dtype: Optional[torch.dtype] = None,
                     device: Optional[torch.device] = None):
    """Creates a binary mask for each sequence.

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    """
    assert len(length.shape) == 1

    if max_len is None:
      max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
      max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
      dtype = length.dtype

    if device is None:
      device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=99, n_mels=64, fmin=50, fmax=14000, is_log=True, 
        ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is 
        the pytorch implementation of as librosa.filters.mel 
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)
        
        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output


    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of 
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec

def MyNTXentLoss(anchor, positive, negative, temperature=0.07):
    similarity_matrix_pos = F.cosine_similarity(anchor[None,:,:], positive[:,None,:], dim=-1).diag().unsqueeze(1)
    similarity_matrix_neg = F.cosine_similarity(anchor[None,:,:], negative[:,None,:], dim=-1)
    logits = torch.cat([similarity_matrix_pos, similarity_matrix_neg], dim=1) / temperature
    
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1
    
    return F.cross_entropy(logits, labels, reduction="mean")

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class ElasticArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.40, std=0.03, plus=True):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std=std
        self.plus=plus
    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.long().view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label.long()[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label.long()[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, elastic_margin=False, elastic_margin_params=(0.4, 0.0125)):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.elastic_margin = elastic_margin
        self.elastic_margin_params = elastic_margin_params
        #

    def forward(self, input, label, device="cuda"):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        if self.elastic_margin:
            index = torch.where(label != -1)[0]
            m_hot = torch.zeros(cosine.size()[0], cosine.size()[1], device=cosine.device)
            margin = torch.normal(mean=self.elastic_margin_params[0], std=self.elastic_margin_params[1], size=label[index, None].size(), device=cosine.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
            with torch.no_grad():
                distmat = cosine[index, label.long().view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label.long()[index, None], margin[idicate_cosie])        
            cos_m = torch.cos(m_hot)
            sin_m = torch.sin(m_hot)
            phi = cosine * cos_m - sine * sin_m
        else:
            phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class AttentionPooling(nn.Module):
    def __init__(self, embedding_size, scale_attn=False):
        super().__init__()
        self.scale_attn = scale_attn
        self.attn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1)
        )

    def forward(self, x, mask=None):
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -float('inf')
        if self.scale_attn:
            scale_factor = 1 / math.sqrt(x.size(-1))
            attn_logits *= scale_factor
        attn_weights = torch.softmax(attn_logits, dim=1)
        x = x * attn_weights
        x = x.sum(dim=1)
        return x

class FeedForward(nn.Module):
    def __init__(self, emb_dim=768, mult=4, p=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mult),
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(emb_dim * mult, emb_dim)
        )

    def forward(self, x):
        return self.fc(x)

class RankConformerModel(nn.Module):
    def __init__(
        self,
        max_len: int = 100,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        cls="arcface",
    ) -> None:

        super(RankConformerModel, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        self.bn0 = nn.BatchNorm1d(input_dim)
        
        self.conformer = Conformer(
            input_dim=emb_dim,
            num_heads=num_heads,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.pooling = AttentionPooling(emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = cls
        
        # self.output = ArcMarginProduct(emb_dim, num_classes, s=64, m=0.4, easy_margin=False, elastic_margin=True)
        self.output = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        length = x.shape[-1]
        if randomized:
            lenghts = torch.randint(10, 50, (x.shape[0],)).to(x.device)
            lenghts[0] = 50
        else:
            lenghts = torch.tensor([length] * x.shape[0]).to(x.device)

        # x = self.bn0(x)
        x = x.transpose(2, 1)
        x = self.proj(x)
        x = self.conformer(x, lenghts)[0]
        x = self.ln(x)
        mask = None
        if randomized:
            mask = torch.zeros(x.shape[0], x.shape[1])
            for i, l in enumerate(lenghts):
                mask[i, l.item():] = 1
            mask = mask.bool().to(x.device)

        f_c = self.pooling(x, mask=mask)
        cls = self.output(self.bn(f_c))
        
        return dict(f_c=f_c, cls=cls)

class ConformerModel(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        cls="arcface",
    ) -> None:

        super(ConformerModel, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        self.bn0 = nn.BatchNorm1d(input_dim)
        
        self.conformer = Conformer(
            input_dim=emb_dim,
            num_heads=num_heads,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.pooling = AttentionPooling(emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = cls
        
        # self.output = ArcMarginProduct(emb_dim, num_classes, s=64, m=0.4, easy_margin=False, elastic_margin=True)
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        length = x.shape[-1]
        if randomized:
            lenghts = torch.randint(10, 50, (x.shape[0],)).to(x.device)
            lenghts[0] = 50
        else:
            lenghts = torch.tensor([length] * x.shape[0]).to(x.device)

        # x = self.bn0(x)
        x = x.transpose(2, 1)
        x = self.proj(x)
        x = self.conformer(x, lenghts)[0]
        x = self.ln(x)
        mask = None
        if randomized:
            mask = torch.zeros(x.shape[0], x.shape[1])
            for i, l in enumerate(lenghts):
                mask[i, l.item():] = 1
            mask = mask.bool().to(x.device)

        if self.cls == "arcface":
            f_c = self.pooling(x, mask=mask)
    
            if pretrain:
                return dict(f_c=x, cls=x)
                
            if labels is None:
                return dict(f_c=f_c, cls=f_c)
                
            cls = self.output(self.bn(f_c), labels)
            return dict(f_c=f_c, cls=cls)
        else:
            cls = self.output(x)
            return dict(f_c=cls, cls=cls)

class ChunkConformerModel(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        cls="arcface",
    ) -> None:

        super(ChunkConformerModel, self).__init__()

        # self.proj = nn.Sequential(
        #     torch.nn.Conv1d(in_channels=input_dim, out_channels=emb_dim//2, kernel_size=3, padding=1),
        #     nn.Dropout(dropout),
        #     nn.GELU(),
        #     torch.nn.Conv1d(in_channels=emb_dim//2, out_channels=emb_dim, kernel_size=7, padding=3)
        # )
            
        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        self.bn0 = nn.BatchNorm1d(input_dim)
        
        self.conformer = Conformer(
            input_dim=emb_dim,
            num_heads=num_heads,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.pooling = AttentionPooling(emb_dim)
        self.chunk_pooling = AttentionPooling(emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = cls
        self.crop_size = 10
        self.step_size = 5
        self.num_steps = int((max_len - self.crop_size) / self.step_size) + 1
        
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        length = x.shape[-1]
        lenghts = torch.tensor([length] * x.shape[0]).to(x.device)

        chunk_embeds = []
        mask = None
        for step in range(self.num_steps):
            inp = x[:, :, self.step_size*step : self.crop_size+self.step_size*step]
            length = inp.shape[-1]
            lenghts = torch.tensor([length] * x.shape[0]).to(x.device)
            inp = inp.transpose(2, 1)
            inp = self.proj(inp)
            inp = self.conformer(inp, lenghts)[0]
            inp = self.ln(inp)
            f_c_chunk = self.pooling(inp, mask=mask)
            chunk_embeds.append(f_c_chunk)

        chunk_x = torch.stack(chunk_embeds, dim=1)
        chunk_x = self.transformer_encoder(chunk_x)
        f_c = self.chunk_pooling(chunk_x, mask)
        
        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class Projector(nn.Module):
    def __init__(self, input_dim=84, emb_dim=1024, mult=2, p=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, emb_dim // mult),
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(emb_dim // mult, emb_dim),
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, x):
        return self.fc(x)

class ChunkConformerOptV5Model(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        crop_size=10,
        step_size=5,
        cls="arcface",
        pos=False,
        scale_attn=False,
        stat_pooling=False
    ) -> None:

        super(ChunkConformerOptV5Model, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )

        self.proj_overall = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.conformer = Conformer(
            input_dim=emb_dim,
            num_heads=num_heads,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )

        self.conformer_overall = Conformer(
            input_dim=emb_dim,
            num_heads=4,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.ln = nn.LayerNorm(emb_dim)
        if stat_pooling:
            self.pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.chunk_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.overall_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
        else:
            self.pooling = AttentionPooling(emb_dim, scale_attn)
            self.chunk_pooling = AttentionPooling(emb_dim, scale_attn)
            self.overall_pooling = AttentionPooling(emb_dim, scale_attn)

        self.pos_enc = nn.Embedding(crop_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
        self.bn = nn.BatchNorm1d(emb_dim*2)
        self.cls = cls
        self.crop_size = crop_size
        self.step_size = step_size
        self.num_steps = int((max_len - self.crop_size) / self.step_size) + 1
        self.pos = pos
        self.stat_pooling = stat_pooling
        
        self.output = ElasticArcFace(emb_dim*2, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        batch_size = x.shape[0]
        reshaped_x = []

        mask = None
        for step in range(self.num_steps):
            inp = x[:, :, self.step_size*step : self.crop_size+self.step_size*step]
            reshaped_x.append(inp.transpose(2, 1))

        reshaped_x = torch.cat(reshaped_x) # bs*num_steps, seq, features
        lenghts = torch.tensor([self.crop_size] * reshaped_x.shape[0]).to(x.device)
        reshaped_x = self.proj(reshaped_x)
        reshaped_x = self.conformer(reshaped_x, lenghts)[0]
        reshaped_x = self.ln(reshaped_x) 
        if self.stat_pooling:
            f_c_chunk = self.pooling(reshaped_x) # bs*num_steps, seq, features -> # bs*num_steps, features
        else:
            f_c_chunk = self.pooling(reshaped_x, mask)

        chunk_embeds = []
        for step in range(self.num_steps):
            chunk_embeds.append(f_c_chunk[batch_size * step : batch_size * (step + 1)])

        chunk_x = torch.stack(chunk_embeds, dim=1)

        if self.pos:
            pos = torch.arange(self.num_steps).repeat(batch_size, 1).to(x.device)
            chunk_x += self.pos_enc(pos)
        
        chunk_x = self.transformer_encoder(chunk_x)
        if self.stat_pooling:
            f_c = self.chunk_pooling(chunk_x)
        else:
            f_c = self.chunk_pooling(chunk_x, mask)

        length = x.shape[-1]
        lenghts = torch.tensor([length] * x.shape[0]).to(x.device)

        x = x.transpose(2, 1)
        x = self.proj_overall(x)
        x = self.conformer_overall(x, lenghts)[0]
        f_c_overall = self.overall_pooling(x, mask)
        # f_c += f_c_overall
        f_c = torch.cat([f_c, f_c_overall], dim=1)
        
        if labels is None:
            return dict(f_c=f_c, cls=f_c)

        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)


class ChunkConformerOptModel(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        crop_size=10,
        step_size=5,
        cls="arcface",
        pos=False,
        scale_attn=False,
        stat_pooling=False
    ) -> None:

        super(ChunkConformerOptModel, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.conformer = Conformer(
            input_dim=emb_dim,
            num_heads=num_heads,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.ln = nn.LayerNorm(emb_dim)
        if stat_pooling:
            self.pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.chunk_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
        else:
            self.pooling = AttentionPooling(emb_dim, scale_attn)
            self.chunk_pooling = AttentionPooling(emb_dim, scale_attn)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = cls
        self.crop_size = crop_size
        self.step_size = step_size
        self.num_steps = int((max_len - self.crop_size) / self.step_size) + 1
        self.pos = pos
        self.stat_pooling = stat_pooling
        if pos:
            self.pos_enc = nn.Embedding(crop_size, emb_dim)
        
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        batch_size = x.shape[0]
        reshaped_x = []

        mask = None
        for step in range(self.num_steps):
            inp = x[:, :, self.step_size*step : self.crop_size+self.step_size*step]
            reshaped_x.append(inp.transpose(2, 1))

        reshaped_x = torch.cat(reshaped_x) # bs*num_steps, seq, features
        reshaped_x = self.proj(reshaped_x)
        lenghts = torch.tensor([self.crop_size] * reshaped_x.shape[0]).to(x.device)
        reshaped_x = self.conformer(reshaped_x, lenghts)[0]
        reshaped_x = self.ln(reshaped_x) 
        if self.stat_pooling:
            f_c_chunk = self.pooling(reshaped_x) # bs*num_steps, seq, features -> # bs*num_steps, features
        else:
            f_c_chunk = self.pooling(reshaped_x, mask)

        chunk_embeds = []
        for step in range(self.num_steps):
            chunk_embeds.append(f_c_chunk[batch_size * step : batch_size * (step + 1)])

        chunk_x = torch.stack(chunk_embeds, dim=1)

        if self.pos:
            pos = torch.arange(self.num_steps).repeat(batch_size, 1).to(x.device)
            chunk_x += self.pos_enc(pos)
        
        chunk_x = self.transformer_encoder(chunk_x)
        if self.stat_pooling:
            f_c = self.chunk_pooling(chunk_x)
        else:
            f_c = self.chunk_pooling(chunk_x, mask)

        if labels is None:
            return dict(f_c=f_c, cls=f_c)

        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

import torchvision
class ChunkConformerOptV6Model(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        crop_size=10,
        step_size=5,
        cls="arcface",
        pos=False,
        scale_attn=False,
        stat_pooling=False
    ) -> None:

        super(ChunkConformerOptV6Model, self).__init__()

        self.base_model = torchvision.models.densenet121(pretrained=True).features
        self.base_model[0] = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.up = torch.nn.Upsample(scale_factor=3, mode='linear')

        self.ln = nn.LayerNorm(emb_dim*2)
        if stat_pooling:
            self.pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.chunk_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
        else:
            self.pooling = AttentionPooling(emb_dim, scale_attn)
            self.chunk_pooling = AttentionPooling(emb_dim*2, scale_attn)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim*2, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
        self.bn = nn.BatchNorm1d(emb_dim*2)
        self.cls = cls
        self.crop_size = crop_size
        self.step_size = step_size
        self.num_steps = int((max_len - self.crop_size) / self.step_size) + 1
        self.pos = pos
        self.stat_pooling = stat_pooling
        if pos:
            self.pos_enc = nn.Embedding(crop_size, emb_dim)
        
        self.output = ElasticArcFace(emb_dim*2, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        batch_size = x.shape[0]
        reshaped_x = []

        mask = None
        for step in range(self.num_steps):
            inp = x[:, :, self.step_size*step : self.crop_size+self.step_size*step]
            reshaped_x.append(self.up(inp).transpose(2, 1))

        reshaped_x = torch.cat(reshaped_x) # bs*num_steps, seq, features
        reshaped_x = self.base_model(reshaped_x.unsqueeze(1)).reshape(reshaped_x.shape[0], -1)
        reshaped_x = self.ln(reshaped_x) 

        chunk_embeds = []
        for step in range(self.num_steps):
            chunk_embeds.append(reshaped_x[batch_size * step : batch_size * (step + 1)])

        chunk_x = torch.stack(chunk_embeds, dim=1)
        chunk_x = self.transformer_encoder(chunk_x)
        if self.stat_pooling:
            f_c = self.chunk_pooling(chunk_x)
        else:
            f_c = self.chunk_pooling(chunk_x, mask)

        if labels is None:
            return dict(f_c=f_c, cls=f_c)

        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class ChunkConformerOptV3Model(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        crop_size=10,
        step_size=5,
        cls="arcface",
        pos=False,
        scale_attn=False,
        stat_pooling=False
    ) -> None:

        super(ChunkConformerOptV3Model, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )

        self.proj_overall = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.conformer = Conformer(
            input_dim=emb_dim,
            num_heads=num_heads,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )

        self.conformer_overall = Conformer(
            input_dim=input_dim,
            num_heads=4,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.ln = nn.LayerNorm(emb_dim)
        if stat_pooling:
            self.pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.chunk_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.overall_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
        else:
            self.pooling = AttentionPooling(emb_dim, scale_attn)
            self.chunk_pooling = AttentionPooling(emb_dim, scale_attn)
            self.overall_pooling = AttentionPooling(emb_dim, scale_attn)

        self.pos_enc = nn.Embedding(crop_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = cls
        self.crop_size = crop_size
        self.step_size = step_size
        self.num_steps = int((max_len - self.crop_size) / self.step_size) + 1
        self.pos = pos
        self.stat_pooling = stat_pooling
        
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        batch_size = x.shape[0]
        reshaped_x = []

        mask = None
        for step in range(self.num_steps):
            inp = x[:, :, self.step_size*step : self.crop_size+self.step_size*step]
            reshaped_x.append(inp.transpose(2, 1))

        reshaped_x = torch.cat(reshaped_x) # bs*num_steps, seq, features
        lenghts = torch.tensor([self.crop_size] * reshaped_x.shape[0]).to(x.device)
        reshaped_x = self.proj(reshaped_x)
        reshaped_x = self.conformer(reshaped_x, lenghts)[0]
        reshaped_x = self.ln(reshaped_x) 
        if self.stat_pooling:
            f_c_chunk = self.pooling(reshaped_x) # bs*num_steps, seq, features -> # bs*num_steps, features
        else:
            f_c_chunk = self.pooling(reshaped_x, mask)

        chunk_embeds = []
        for step in range(self.num_steps):
            chunk_embeds.append(f_c_chunk[batch_size * step : batch_size * (step + 1)])

        chunk_x = torch.stack(chunk_embeds, dim=1)

        if self.pos:
            pos = torch.arange(self.num_steps).repeat(batch_size, 1).to(x.device)
            chunk_x += self.pos_enc(pos)
        
        chunk_x = self.transformer_encoder(chunk_x)
        if self.stat_pooling:
            f_c = self.chunk_pooling(chunk_x)
        else:
            f_c = self.chunk_pooling(chunk_x, mask)

        length = x.shape[-1]
        lenghts = torch.tensor([length] * x.shape[0]).to(x.device)

        x = x.transpose(2, 1)
        x = self.conformer_overall(x, lenghts)[0]
        x = self.proj_overall(x)
        f_c_overall = self.overall_pooling(x, mask)
        f_c += f_c_overall
        # f_c = torch.cat([f_c, f_c_overall], dim=1)
        
        if labels is None:
            return dict(f_c=f_c, cls=f_c)

        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class ChunkConformerOptV4Model(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        crop_size=10,
        step_size=5,
        cls="arcface",
        pos=False,
        scale_attn=False,
        stat_pooling=False
    ) -> None:

        super(ChunkConformerOptV4Model, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )

        self.proj_overall = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.conformer = Conformer(
            input_dim=emb_dim,
            num_heads=num_heads,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )

        self.conformer_overall = Conformer(
            input_dim=input_dim,
            num_heads=4,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        self.ln = nn.LayerNorm(emb_dim)
        if stat_pooling:
            self.pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.chunk_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.overall_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
        else:
            self.pooling = AttentionPooling(emb_dim, scale_attn)
            self.chunk_pooling = AttentionPooling(emb_dim, scale_attn)
            self.overall_pooling = AttentionPooling(emb_dim, scale_attn)

        self.pos_enc = nn.Embedding(crop_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
        self.bn = nn.BatchNorm1d(emb_dim*2)
        self.cls = cls
        self.crop_size = crop_size
        self.step_size = step_size
        self.num_steps = int((max_len - self.crop_size) / self.step_size) + 1
        self.pos = pos
        self.stat_pooling = stat_pooling
        
        self.output = ElasticArcFace(emb_dim*2, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        batch_size = x.shape[0]
        reshaped_x = []

        mask = None
        for step in range(self.num_steps):
            inp = x[:, :, self.step_size*step : self.crop_size+self.step_size*step]
            reshaped_x.append(inp.transpose(2, 1))

        reshaped_x = torch.cat(reshaped_x) # bs*num_steps, seq, features
        lenghts = torch.tensor([self.crop_size] * reshaped_x.shape[0]).to(x.device)
        reshaped_x = self.proj(reshaped_x)
        reshaped_x = self.conformer(reshaped_x, lenghts)[0]
        reshaped_x = self.ln(reshaped_x) 
        if self.stat_pooling:
            f_c_chunk = self.pooling(reshaped_x) # bs*num_steps, seq, features -> # bs*num_steps, features
        else:
            f_c_chunk = self.pooling(reshaped_x, mask)

        chunk_embeds = []
        for step in range(self.num_steps):
            chunk_embeds.append(f_c_chunk[batch_size * step : batch_size * (step + 1)])

        chunk_x = torch.stack(chunk_embeds, dim=1)

        if self.pos:
            pos = torch.arange(self.num_steps).repeat(batch_size, 1).to(x.device)
            chunk_x += self.pos_enc(pos)
        
        chunk_x = self.transformer_encoder(chunk_x)
        if self.stat_pooling:
            f_c = self.chunk_pooling(chunk_x)
        else:
            f_c = self.chunk_pooling(chunk_x, mask)

        length = x.shape[-1]
        lenghts = torch.tensor([length] * x.shape[0]).to(x.device)

        x = x.transpose(2, 1)
        x = self.conformer_overall(x, lenghts)[0]
        x = self.proj_overall(x)
        f_c_overall = self.overall_pooling(x, mask)
        # f_c += f_c_overall
        f_c = torch.cat([f_c, f_c_overall], dim=1)
        
        if labels is None:
            return dict(f_c=f_c, cls=f_c)

        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)


class ChunkConformerOptV2Model(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        crop_size=10,
        step_size=5,
        cls="arcface",
        pos=False,
        scale_attn=False,
        stat_pooling=False
    ) -> None:

        super(ChunkConformerOptV2Model, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.conformer = Conformer(
            input_dim=emb_dim,
            num_heads=num_heads,
            ffn_dim=512,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )

        # self.chunk_conformer = Conformer(
        #     input_dim=emb_dim,
        #     num_heads=num_heads,
        #     ffn_dim=512,
        #     num_layers=num_layers,
        #     depthwise_conv_kernel_size=31,
        #     dropout=dropout
        # )

        encoder_layer_chunk = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.chunk_transformer_encoder = nn.TransformerEncoder(encoder_layer_chunk, num_layers=num_layers)
        
        self.ln = nn.LayerNorm(emb_dim)
        if stat_pooling:
            self.pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
            self.chunk_pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
        else:
            self.pooling = AttentionPooling(emb_dim, scale_attn)
            self.chunk_pooling = AttentionPooling(emb_dim, scale_attn)
            self.final_pooling = AttentionPooling(emb_dim, scale_attn)

        self.pos_enc = nn.Embedding(crop_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = cls
        self.crop_size = crop_size
        self.step_size = step_size
        self.num_steps = int((max_len - self.crop_size) / self.step_size) + 1
        self.pos = pos
        self.stat_pooling = stat_pooling
        
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        batch_size = x.shape[0]
        reshaped_x = []

        mask = None
        for step in range(self.num_steps):
            inp = x[:, :, self.step_size*step : self.crop_size+self.step_size*step]
            reshaped_x.append(inp.transpose(2, 1))

        reshaped_x = torch.cat(reshaped_x) # bs*num_steps, seq, features
        reshaped_x = self.proj(reshaped_x)
        lenghts = torch.tensor([self.crop_size] * reshaped_x.shape[0]).to(x.device)
        reshaped_x = self.conformer(reshaped_x, lenghts)[0]
        reshaped_x = self.ln(reshaped_x) 
        if self.stat_pooling:
            f_c_chunk = self.pooling(reshaped_x) # bs*num_steps, seq, features -> # bs*num_steps, features
        else:
            f_c_chunk = self.pooling(reshaped_x, mask)

        chunk_embeds = []
        for step in range(self.num_steps):
            chunk_embeds.append(f_c_chunk[batch_size * step : batch_size * (step + 1)])
        chunk_x = torch.stack(chunk_embeds, dim=1) # bs, seq * num_interm_steps, features (32, 45, 1024)

        chunk_x_2 = []

        mask = None
        crop_size_2 = 5
        step_size_2 = 1
        num_steps_2 = 5
        for step in range(num_steps_2):
            inp = chunk_x[:, step_size_2*step : crop_size_2+step_size_2*step, :]
            chunk_x_2.append(inp) #  (32 * 5, 9, 1024)
            
        chunk_x_2 = torch.cat(chunk_x_2)
        lenghts = torch.tensor([crop_size_2] * chunk_x_2.shape[0]).to(x.device)
        # chunk_x_2 = self.chunk_conformer(chunk_x_2, lenghts)[0]
        chunk_x_2 = self.chunk_transformer_encoder(chunk_x_2)
        f_c_chunk_2 = self.chunk_pooling(chunk_x_2, mask)

        chunk_embeds_2 = []
        for step in range(num_steps_2):
            chunk_embeds_2.append(f_c_chunk_2[batch_size * step : batch_size * (step + 1)])
        chunk_x_2 = torch.stack(chunk_embeds_2, dim=1) # bs, seq * num_interm_steps, features (32, 45, 1024)

        f_c_chunk_2 = self.transformer_encoder(chunk_x_2)
        if self.stat_pooling:
            f_c = self.final_pooling(f_c_chunk_2)
        else:
            f_c = self.final_pooling(f_c_chunk_2, mask)

        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class ChunkTransformerCNNOptModel(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        cls="arcface",
    ) -> None:

        super(ChunkTransformerCNNOptModel, self).__init__()


        self.proj = Resnet50(
            Bottleneck,
            num_channels=1,
        )
        self.bn0 = nn.BatchNorm1d(input_dim)
        
        self.ln = nn.LayerNorm(emb_dim)
        self.pooling = AttentionPooling(emb_dim)
        self.chunk_pooling = AttentionPooling(emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.bn = nn.BatchNorm1d(emb_dim)
        self.cls = cls
        self.crop_size = 10
        self.step_size = 5
        self.num_steps = int((max_len - self.crop_size) / self.step_size) + 1
        
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):     
        
        batch_size = x.shape[0]
        reshaped_x = []
        
        mask = None
        for step in range(self.num_steps):
            inp = x[:, :, self.step_size*step : self.crop_size+self.step_size*step]
            reshaped_x.append(inp.transpose(2, 1))

        reshaped_x = torch.cat(reshaped_x) # bs*num_steps, seq, features
        reshaped_x = self.proj(reshaped_x)
        reshaped_x = reshaped_x[:, :, 0]
        reshaped_x = reshaped_x.transpose(2, 1)
        reshaped_x = self.ln(reshaped_x) 
        f_c_chunk = self.pooling(reshaped_x, mask=mask)        

        chunk_embeds = []
        for step in range(self.num_steps):
            chunk_embeds.append(f_c_chunk[batch_size * step : batch_size * (step + 1)])
        chunk_x = torch.stack(chunk_embeds, dim=1)
        chunk_x = self.transformer_encoder(chunk_x)
        f_c = self.chunk_pooling(chunk_x, mask)

        # print(y)
        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class ConformerModelMoE(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        num_branches=2
    ) -> None:

        super(ConformerModelMoE, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim * 4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        # ast_config = ASTConfig(num_mel_bins=emb_dim, max_length=max_len, num_hidden_layers=num_layers, hidden_size=emb_dim, num_attention_heads=num_heads)

        self.moe1 = nn.ParameterList([Conformer(input_dim=emb_dim,
                                               num_heads=num_heads,
                                               ffn_dim=512,
                                               num_layers=num_layers,
                                               depthwise_conv_kernel_size=31,
                                               dropout=dropout) for _ in range(num_branches)]) 
        # self.moe2 = nn.ParameterList([ASTModel(ast_config) for _ in range(num_branches)]) 
    
        self.ln = nn.LayerNorm(emb_dim)
        self.pooling = AttentionPooling(emb_dim)
        # self.pooling = AttentiveStatisticsPooling(emb_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        
        # self.output = ArcMarginProduct(emb_dim, num_classes, s=64, m=0.4, easy_margin=False, elastic_margin=True)
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):
        length = x.shape[-1]
        lenghts = torch.tensor([length] * x.shape[0]).to(x.device)
        
        x = x.transpose(2, 1)
        x = self.proj(x)
        
        prediction = None
        for branch in self.moe1:
            if prediction is None:
                prediction = branch(x, lenghts)[0]
            else:
                prediction += branch(x, lenghts)[0]
                
        # x = self.conformer(x, lenghts)[0]
        x = self.ln(prediction)
        f_c = self.pooling(x, mask=None)

        # for branch in self.moe2:
        #      f_c += branch(x).last_hidden_state.mean(1)
        
        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class TransformerModelMoE(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        num_branches=4
    ) -> None:

        super(TransformerModelMoE, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim * 4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        # ast_config = ASTConfig(num_mel_bins=emb_dim, max_length=max_len, num_hidden_layers=num_layers, hidden_size=emb_dim, num_attention_heads=num_heads)

        self.moe1 = nn.ParameterList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads), num_layers=num_layers) for _ in range(num_branches)]) 
        # self.moe2 = nn.ParameterList([ASTModel(ast_config) for _ in range(num_branches)]) 
    
        self.ln = nn.LayerNorm(emb_dim)
        self.pooling = AttentionPooling(emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        
        # self.output = ArcMarginProduct(emb_dim, num_classes, s=64, m=0.4, easy_margin=False, elastic_margin=True)
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None):
        length = x.shape[-1]
        lenghts = torch.tensor([length] * x.shape[0]).to(x.device)
        
        x = x.transpose(2, 1)
        x = self.proj(x)
        
        prediction = None
        for branch in self.moe1:
            if prediction is None:
                prediction = branch(x)
            else:
                prediction += branch(x)
                
        x = self.ln(prediction)
        f_c = self.pooling(x, mask=None)

        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class TransformerFreqModel(nn.Module):
    def __init__(
        self,
        max_len: int = 50,
        input_dim: int = 84,
        emb_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 10,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84,
        num_branches=4
    ) -> None:

        super(TransformerFreqModel, self).__init__()
        
        # self.embeddings = nn.Embedding(input_dim, emb_dim)
        self.trasnformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim, nhead=4), num_layers=num_layers)
    
        self.ln = nn.LayerNorm(input_dim)
        self.pooling = AttentionPooling(input_dim, False)
        self.bn = nn.BatchNorm1d(input_dim)
        
        self.output = ElasticArcFace(input_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False, pretrain=False):        
        x = x.transpose(2, 1)
        # x = x.argmax(-1)
        
        # x = self.embeddings(x)
        x = self.trasnformer(x)
        x = self.ln(x)
        f_c = self.pooling(x, mask=None)

        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)


class CNNModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1000,
        max_len: int = 50,
        input_dim: int = 84,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84
    ) -> None:

        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.resnet = resnet50()
        self.bn = nn.BatchNorm1d(emb_dim)
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None):
        x = self.conv1(x[:, None])
        f_c = self.resnet(x)

        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class EffNetModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1280,
        max_len: int = 50,
        input_dim: int = 84,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84
    ) -> None:

        super(EffNetModel, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False
        )
        # self.eff_net_model = timm.create_model("hf_hub:timm/eca_nfnet_l0", pretrained=True)#EfficientNetModel.from_pretrained("google/efficientnet-b0")
        self.eff_net_model = timm.create_model(
                'efficientnet_b0', pretrained=True,
                num_classes=1, in_chans=1,
            )
        self.eff_net_model.classifier = torch.nn.Identity()
        self.bn = nn.BatchNorm1d(emb_dim)
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False):
        # x = self.conv1(x[:, None])
        # f_c = self.eff_net_model(x).pooler_output
        f_c = self.eff_net_model(x[:, None])

        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)

class AST_Model(nn.Module):
    def __init__(
        self,
        emb_dim: int = 768,
        max_len: int = 50,
        input_dim: int = 84,
        num_classes: int = 8858,
        dropout=0.1,
        n_bins=84
    ) -> None:

        super(AST_Model, self).__init__()
        
        ast_config = ASTConfig(num_mel_bins=input_dim, max_length=max_len)
        self.ast_model = ASTModel(ast_config)
        self.pooling = AttentionPooling(emb_dim)
        self.bn0 = nn.BatchNorm1d(input_dim)
        
        self.bn = nn.BatchNorm1d(emb_dim)
        self.output = ElasticArcFace(emb_dim, num_classes, plus=True)

    def forward(self, x: torch.Tensor, labels=None, randomized=False):
        x = self.bn0(x)
        x = x.transpose(2, 1)
        x = self.ast_model(x).last_hidden_state
        f_c = self.pooling(x)

        if labels is None:
            return dict(f_c=f_c, cls=f_c)
            
        cls = self.output(self.bn(f_c), labels)
        return dict(f_c=f_c, cls=cls)