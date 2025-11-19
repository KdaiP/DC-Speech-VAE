"""
modified from:
https://github.com/yxlllc/DDSP-SVC/blob/6.3/optimizer/muon.py
which support convolution
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def exists(val):
    return val is not None

# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315

def zeropower_via_newtonschulz5(G: Tensor, steps: int, use_bf16: bool = False) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim == 3 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (DEFAULT_A, DEFAULT_B,  DEFAULT_C)
    if use_bf16:
        X = G.bfloat16()
    else:
        X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = F.normalize(X, p=2.0, dim=(-2, -1), eps=1e-7)
    
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1)
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=1e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            shape_groups = {}
            for p in filter(lambda p: exists(p.grad), group['params']):       
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                buf: Tensor = state["momentum_buffer"]     
                key = (p.shape, p.device, p.dtype)
                if key not in shape_groups:
                    shape_groups[key] = {"params": [], "grads": [], "buffers": []}
                shape_groups[key]['params'].append(p)
                shape_groups[key]['grads'].append(g)
                shape_groups[key]['buffers'].append(buf)                
            for key in shape_groups:
                group_data = shape_groups[key]
                g = torch.stack(group_data['grads'])
                buf = torch.stack(group_data['buffers'])
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                if g.ndim >= 4: # for the case of conv filters
                    g = g.view(g.size(0), g.size(1), -1)
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                for i, p in enumerate(group_data['params']):
                    if group["weight_decay"] > 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    p.data.add_(g[i].view_as(p), alpha=-group["lr"] * max(g[i].size()) ** 0.5)
                    self.state[p]['momentum_buffer'] = buf[i].clone()

        return loss

class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers

        # PyTorch scheduler 需要 param_groups 属性，这里组合所有 param_groups
        param_groups = []
        for optimizer in optimizers:
            param_groups.extend(optimizer.param_groups)

        self.param_groups = param_groups

    def step(self, closure=None):
        loss = None
        for optimizer in self.optimizers:
            loss = optimizer.step(closure) if closure else optimizer.step()
        return loss

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            f'optimizer_{i}': optimizer.state_dict()
            for i, optimizer in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict):
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(state_dict[f'optimizer_{i}'])

def get_muon_optimizer(models, lr=1e-4, wd=0.1):

    muon_params = []
    adamw_params = []

    for model in models:
        for name, module in model.named_modules():
            for pname, p in module.named_parameters(recurse=False):
                if p.requires_grad:
                    if p.ndim >= 2 and not isinstance(module, nn.Embedding):
                        muon_params.append(p)
                    else:
                        adamw_params.append(p)
                else:
                    print(f'optimizer skip: {name}:{pname}')

    print('Muon params:', len(muon_params))
    print('AdamW params:', len(adamw_params))

    muon_optimizer = Muon(muon_params, lr ,wd)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr, (0.9, 0.95), weight_decay=wd)

    combined_optimizer = CombinedOptimizer([muon_optimizer, adamw_optimizer])

    return combined_optimizer


if __name__ == "__main__":
    model1 = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
    )
    model2 = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
    )

    # 初始化优化器
    optimizer = get_muon_optimizer([model1, model2])

    # scheduler测试
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # 随机输入
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)

    # 训练循环
    for epoch in range(10):
        optimizer.zero_grad()

        outputs1 = model1(inputs)
        output2 = model2(inputs)

        outputs = (outputs1 + output2) / 2
        loss = ((outputs - targets) ** 2).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()}')

    # 测试保存和加载
    print("\nTesting save and load...")
    torch.save(optimizer.state_dict(), 'test_optimizer.pt')

    # 创建新优化器并加载状态
    optimizer.load_state_dict(torch.load('test_optimizer.pt'))

    print("Optimizer state loaded successfully!")
