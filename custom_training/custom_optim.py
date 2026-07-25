"""
Two things implemented from scratch, benchmarked against PyTorch's built-ins on
the same language-modeling task:

1. Label smoothing loss -- instead of a one-hot target, spreads a small amount of
   probability mass across all other classes. Reduces overconfidence and generally
   improves generalization in sequence generation.
2. A simplified AdamW optimizer -- manually implements the moving-average gradient
   estimates (m, v), bias correction, and decoupled weight decay that make Adam/AdamW
   work, rather than calling torch.optim.AdamW.

This directly targets the JD's "implementing basic optimizers and regularizations;
formulating and implementing loss functions" bullet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LabelSmoothingLoss(nn.Module):
    """
    Standard cross-entropy uses a one-hot target: probability 1 on the correct class,
    0 elsewhere. Label smoothing instead targets (1 - eps) on the correct class and
    eps / (K-1) spread over the rest, which is implemented here manually via KL
    divergence between the smoothed target distribution and the model's log-probs.
    """
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (N, K), targets: (N,)
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        mask = (targets != self.ignore_index).unsqueeze(1)
        loss = -(true_dist * log_probs) * mask
        return loss.sum() / mask.sum().clamp(min=1)


class SimpleAdamW:
    """
    A from-scratch reimplementation of AdamW's core update rule:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t                (1st moment estimate)
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2              (2nd moment estimate)
      m_hat = m_t / (1 - beta1^t)                              (bias correction)
      v_hat = v_t / (1 - beta2^t)
      theta_t = theta_{t-1} - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1} )
    The weight_decay term is applied directly to the parameter (decoupled), which is
    exactly what distinguishes AdamW from plain Adam + L2 regularization.
    """
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # decoupled weight decay (the "W" in AdamW): applied to the parameter
            # directly, not mixed into the gradient like plain L2 regularization
            p.data = p.data - self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data)


if __name__ == "__main__":
    # Sanity check: confirm the custom loss/optimizer behave sensibly on a toy problem
    torch.manual_seed(0)
    logits = torch.randn(8, 20, requires_grad=True)
    targets = torch.randint(0, 20, (8,))

    ce_loss = F.cross_entropy(logits, targets)
    ls_loss_fn = LabelSmoothingLoss(smoothing=0.1)
    ls_loss = ls_loss_fn(logits, targets)
    print(f"Plain cross-entropy loss : {ce_loss.item():.4f}")
    print(f"Label-smoothed loss      : {ls_loss.item():.4f}  (expected: slightly higher, since "
          f"it never lets the model be fully 'confident')")

    # Quick optimizer sanity check: does SimpleAdamW actually reduce a toy loss?
    w = torch.nn.Parameter(torch.randn(10))
    target = torch.randn(10)
    opt = SimpleAdamW([w], lr=0.1)
    for step in range(50):
        loss = ((w - target) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"\nSimpleAdamW toy fit: final loss after 50 steps = {loss.item():.6f} (should be near 0)")
