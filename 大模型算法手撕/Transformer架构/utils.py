"""
工具函数模块
提供一些辅助函数，用于模型训练和推理
"""

import torch
import torch.nn as nn


def count_parameters(model):
    """
    统计模型参数数量
    Args:
        model: PyTorch 模型
    Returns:
        total: 总参数量
        trainable: 可训练参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_padding_mask(seq, pad_idx=0):
    """
    创建填充掩码
    Args:
        seq: 序列张量 [batch_size, seq_len]
        pad_idx: 填充标记的索引
    Returns:
        mask: 填充掩码 [batch_size, 1, 1, seq_len]
    """
    # 标记填充位置
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """
    创建前瞻掩码（用于解码器的自注意力）
    Args:
        size: 序列长度
    Returns:
        mask: 前瞻掩码 [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask


def create_masks(src, tgt, pad_idx=0):
    """
    创建所有需要的掩码
    Args:
        src: 源序列 [batch_size, src_len]
        tgt: 目标序列 [batch_size, tgt_len]
        pad_idx: 填充标记的索引
    Returns:
        src_mask: 源序列掩码
        tgt_mask: 目标序列掩码（结合填充和前瞻掩码）
    """
    # 源序列填充掩码
    src_mask = create_padding_mask(src, pad_idx)

    # 目标序列填充掩码
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)

    # 目标序列前瞻掩码
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)
    tgt_look_ahead_mask = tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)

    # 组合目标序列的两种掩码
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask

    return src_mask, tgt_mask


def initialize_weights(model):
    """
    初始化模型权重
    Args:
        model: PyTorch 模型
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class LabelSmoothing(nn.Module):
    """
    标签平滑
    减少模型过度自信，提高泛化能力
    """

    def __init__(self, size, padding_idx, smoothing=0.1):
        """
        Args:
            size: 词汇表大小
            padding_idx: 填充标记索引
            smoothing: 平滑系数
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        Args:
            x: 模型输出 [batch_size * seq_len, vocab_size]
            target: 目标标签 [batch_size * seq_len]
        Returns:
            loss: 平滑后的损失
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    """
    学习率调度器（Noam 调度）
    实现论文中的学习率预热和衰减策略
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        """
        Args:
            model_size: 模型维度
            factor: 缩放因子
            warmup: 预热步数
            optimizer: 优化器
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        更新学习率并执行一步优化
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        计算当前学习率
        """
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    贪婪解码
    Args:
        model: Transformer 模型
        src: 源序列 [batch_size, src_len]
        src_mask: 源序列掩码
        max_len: 最大解码长度
        start_symbol: 起始标记
    Returns:
        decoded: 解码结果 [batch_size, max_len]
    """
    batch_size = src.size(0)

    # 编码
    encoder_output = model.encode(src, src_mask)

    # 初始化解码器输入（起始标记）
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)

    # 逐步解码
    for i in range(max_len - 1):
        # 解码
        out = model.decode(ys, encoder_output, None, src_mask)

        # 获取下一个词的概率分布
        prob = out[:, -1, :]

        # 选择概率最大的词
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)

        # 添加到解码序列
        ys = torch.cat([ys, next_word], dim=1)

    return ys


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    保存模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """
    加载模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径
    Returns:
        epoch: 轮次
        loss: 损失
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}")
    return epoch, loss
