import torch
import torch.nn.functional as F


class LinfPGDAttack:
    def __init__(self, model, loss_fn, eps, step_size, num_steps, random_start=True, bounds=(0.0, 1.0)):
        self.model = model
        self.loss_fn = loss_fn

        self.eps = eps
        self.step_size = step_size
        self.num_steps = num_steps
        self.bounds = bounds

        self.random_start = random_start

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @torch.no_grad()
    def perturb(self, x, y):
        delta = torch.zeros_like(x, dtype=self.dtype, device=self.device)
        if self.random_start:
            delta = delta.uniform_(-self.eps, self.eps)
            delta = (x + delta).clamp(*self.bounds) - x

        for _ in range(self.num_steps):
            with torch.enable_grad():
                delta.requires_grad = True
                loss = self.loss_fn(self.model(x + delta), y)
                grads = torch.autograd.grad(loss, delta)[0]
            delta = delta + self.step_size * torch.sign(grads)
            delta = delta.clamp(-self.eps, self.eps)
            delta = (x + delta).clamp(*self.bounds) - x
        return x + delta


# FGSM attack with random start and GradAlign regularization.
class LinfFGSMAttack:
    def __init__(self, model, loss_fn, eps, step_size, bounds=(0.0, 1.0)):
        self.model = model
        self.loss_fn = loss_fn

        self.eps = eps
        self.step_size = step_size
        self.bounds = bounds

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @torch.no_grad()
    def perturb(self, x, y):
        delta = torch.zeros_like(x, dtype=self.dtype, device=self.device).uniform_(-self.eps, self.eps)
        delta = (x + delta).clamp(*self.bounds) - x

        delta1 = torch.zeros_like(x, dtype=self.dtype, device=self.device)
        delta2 = delta.clone()
        with torch.enable_grad():
            delta1.requires_grad = True
            delta2.requires_grad = True
            grads1 = torch.autograd.grad(self.loss_fn(self.model(x + delta1), y), delta1)[0].detach()
            grads2 = torch.autograd.grad(self.loss_fn(self.model(x + delta2), y), delta2)[0].detach()

            delta.requires_grad = True
            loss = self.loss_fn(self.model(x + delta), y) + \
                   (1. - F.cosine_similarity(grads1.view(-1), grads2.view(-1), dim=0))
            grads = torch.autograd.grad(loss, delta)[0]
        delta = delta + self.step_size * torch.sign(grads)
        delta = delta.clamp(-self.eps, self.eps)
        return (x + delta).clamp(*self.bounds)
