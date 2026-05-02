"""Self-Supervised Tree-HyperLISTA: test-time adaptation with no training data.

Adapts (c1, c2, c3) per-sample (or per-batch) at test time using only
measurement consistency loss: ||y - A @ x_hat||^2. No ground truth needed.
"""

import torch
import torch.nn as nn
import numpy as np
from src.models.tree_hyperlista import TreeHyperLISTA


class SelfSupervisedTreeHyperLISTA(nn.Module):
    """Tree-HyperLISTA with self-supervised test-time adaptation.

    At test time, for each batch of measurements y:
      1. Initialize (c1, c2, c3) from defaults or pre-tuned values
      2. Run forward pass: x_hat = TreeHyperLISTA(y; c1, c2, c3)
      3. Compute loss = ||y - A @ x_hat||^2  (measurement consistency)
      4. Update (c1, c2, c3) via gradient descent
      5. Repeat for a few steps
      6. Return final x_hat

    Only 3 parameters to optimize -> converges fast, no overfitting.
    """

    def __init__(self, A: np.ndarray, tree_info: dict, num_layers: int = 16,
                 c1_init: float = 1.0, c2_init: float = 0.0, c3_init: float = 3.0,
                 support_mode: str = 'hybrid_tree', rho: float = 0.5,
                 adapt_steps: int = 10, adapt_lr: float = 0.05,
                 num_restarts: int = 1):
        super().__init__()
        self.A_np = A
        self.tree_info = tree_info
        self.num_layers = num_layers
        self.c1_init = c1_init
        self.c2_init = c2_init
        self.c3_init = c3_init
        self.support_mode = support_mode
        self.rho = rho
        self.adapt_steps = adapt_steps
        self.adapt_lr = adapt_lr
        self.num_restarts = num_restarts
        self.num_hyperparams = 3

        self.register_buffer('A', torch.from_numpy(A).float())

    def _create_model(self, c1, c2, c3, device):
        """Create a fresh TreeHyperLISTA with given hyperparameters."""
        model = TreeHyperLISTA(
            self.A_np, self.tree_info,
            num_layers=self.num_layers,
            c1=c1, c2=c2, c3=c3,
            support_mode=self.support_mode,
            rho=self.rho
        ).to(device)
        return model

    def _measurement_loss(self, x_hat, y, A):
        """Measurement consistency: ||y - A @ x_hat||^2."""
        residual = y - x_hat @ A.t()
        return (residual ** 2).sum(dim=-1).mean()

    def _adapt(self, y, c1_start, c2_start, c3_start):
        """Run test-time adaptation. Forces gradients enabled."""
        device = y.device

        model = self._create_model(c1_start, c2_start, c3_start, device)
        A = model.A  # use the model's A buffer

        optimizer = torch.optim.Adam([model.c1, model.c2, model.c3],
                                     lr=self.adapt_lr)

        best_loss = float('inf')
        best_state = None

        # Force gradients on even if called inside torch.no_grad()
        with torch.enable_grad():
            for step in range(self.adapt_steps):
                optimizer.zero_grad()
                x_hat = model(y)
                loss = self._measurement_loss(x_hat, y, A)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {
                        'c1': model.c1.item(),
                        'c2': model.c2.item(),
                        'c3': model.c3.item(),
                    }

                loss.backward()
                optimizer.step()

        if best_state is not None:
            model.c1.data.fill_(best_state['c1'])
            model.c2.data.fill_(best_state['c2'])
            model.c3.data.fill_(best_state['c3'])

        return model, best_loss

    def forward(self, y, return_trajectory=False, num_layers=None):
        device = y.device

        best_overall_loss = float('inf')
        best_model = None

        inits = [(self.c1_init, self.c2_init, self.c3_init)]

        for r in range(self.num_restarts - 1):
            rng = torch.Generator()
            rng.manual_seed(42 + r)
            c1_r = 0.1 + torch.rand(1, generator=rng).item() * 5.0
            c2_r = -4.0 + torch.rand(1, generator=rng).item() * 6.0
            c3_r = 0.1 + torch.rand(1, generator=rng).item() * 5.0
            inits.append((c1_r, c2_r, c3_r))

        for c1_s, c2_s, c3_s in inits:
            model, loss = self._adapt(y, c1_s, c2_s, c3_s)
            if loss < best_overall_loss:
                best_overall_loss = loss
                best_model = model

        best_model.eval()
        with torch.no_grad():
            if return_trajectory:
                return best_model(y, return_trajectory=True, num_layers=num_layers)
            else:
                return best_model(y, num_layers=num_layers)

    def solve(self, y, return_trajectory=False, num_iter=None):
        """Alias for forward, compatible with classical solver interface."""
        return self.forward(y, return_trajectory=return_trajectory,
                           num_layers=num_iter)

    def eval(self):
        return self

    def to(self, device):
        self.A = self.A.to(device)
        return self

    def parameters(self):
        return iter([])


class AmortizedSSTreeHyperLISTA(SelfSupervisedTreeHyperLISTA):
    """Amortized: pre-tune on synthetic data, then fine-tune per-sample."""

    def __init__(self, A: np.ndarray, tree_info: dict, num_layers: int = 16,
                 c1_init: float = 1.0, c2_init: float = 0.0, c3_init: float = 3.0,
                 support_mode: str = 'hybrid_tree', rho: float = 0.5,
                 adapt_steps: int = 7, adapt_lr: float = 0.03,
                 num_restarts: int = 1):
        super().__init__(
            A, tree_info, num_layers,
            c1_init, c2_init, c3_init,
            support_mode, rho,
            adapt_steps, adapt_lr, num_restarts
        )

    def set_pretrained_init(self, c1, c2, c3):
        """Set initialization from pre-tuned values."""
        self.c1_init = c1
        self.c2_init = c2
        self.c3_init = c3
