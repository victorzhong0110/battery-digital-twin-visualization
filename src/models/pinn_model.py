"""Physics-Informed Neural Network (PINN) for battery capacity prediction.

Core Innovation:
    Standard neural networks learn purely from data. PINN embeds physical
    knowledge (the ECM degradation model) directly into the loss function,
    constraining the neural network to respect known battery physics.

Loss = L_data + λ₁·L_physics + λ₂·L_monotone + λ₃·L_boundary

Where:
    L_data:     MSE between predicted and actual capacity
    L_physics:  Violation of ECM-based degradation dynamics
                (dCapacity/dCycle should follow R0/R1 increase pattern)
    L_monotone: Penalize non-monotonic capacity increases (capacity should
                generally decrease with cycling, except for early-life recovery)
    L_boundary: Initial capacity should match rated capacity;
                capacity should never go below 0 or above rated

This bridges the digital twin (Phase 2) and ML prediction (Phase 3):
the ECM parameters calibrated in Phase 2 are used as physics priors here.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PhysicsConstraints:
    """Encapsulates the physical constraints from ECM calibration.

    These come from the ECM parameters (Phase 2) and encode domain knowledge:
    - R0 increases linearly with cycle → capacity drops
    - The relationship: capacity ≈ f(R0, R1, temperature)
    - Capacity is monotonically non-increasing (with some noise tolerance)
    """

    def __init__(
        self,
        r0_initial: float,
        r0_slope: float,
        r1_initial: float,
        r1_slope: float,
        capacity_initial: float,
        capacity_slope: float,
        rated_capacity: float,
    ):
        self.r0_initial = r0_initial
        self.r0_slope = r0_slope
        self.r1_initial = r1_initial
        self.r1_slope = r1_slope
        self.capacity_initial = capacity_initial
        self.capacity_slope = capacity_slope
        self.rated_capacity = rated_capacity

    def expected_capacity_at_cycle(self, cycle: torch.Tensor) -> torch.Tensor:
        """Physics-based capacity estimate: linear degradation model."""
        return self.capacity_initial + self.capacity_slope * cycle

    def expected_resistance_at_cycle(self, cycle: torch.Tensor) -> torch.Tensor:
        """Physics-based total resistance at a given cycle."""
        r0 = self.r0_initial + self.r0_slope * cycle
        r1 = self.r1_initial + self.r1_slope * cycle
        return r0 + r1

    @classmethod
    def from_ecm_params(cls, ecm_params) -> PhysicsConstraints:
        """Create from ECMParams dataclass."""
        return cls(
            r0_initial=ecm_params.r0_initial,
            r0_slope=ecm_params.r0_slope,
            r1_initial=ecm_params.r1_initial,
            r1_slope=ecm_params.r1_slope,
            capacity_initial=ecm_params.capacity_initial,
            capacity_slope=ecm_params.capacity_slope,
            rated_capacity=ecm_params.rated_capacity_ah,
        )


class PINNCapacityPredictor(nn.Module):
    """Physics-Informed Neural Network for capacity prediction.

    Architecture:
        Input features -> Shared encoder -> Two heads:
            1. Data head: directly predicts capacity
            2. Physics head: predicts deviation from physics model
        Final prediction = physics_baseline + learned_residual

    This "residual learning" structure means the network only needs to learn
    what the physics model *can't* explain, making it more data-efficient.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Shared feature encoder
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Residual head: learn the deviation from physics prediction
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh(),  # Bound residual to [-1, 1], scaled later
        )

        # Confidence head: estimate uncertainty of residual
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # Positive uncertainty
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))  # Learnable scale

    def forward(
        self, x: torch.Tensor, physics_baseline: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, input_dim) feature vector
            physics_baseline: (batch, 1) capacity from physics model

        Returns:
            predicted_capacity: (batch, 1)
            uncertainty: (batch, 1)
        """
        h = self.encoder(x)
        residual = self.residual_head(h) * self.residual_scale
        uncertainty = self.confidence_head(h)

        # Final prediction: physics + learned correction
        predicted = physics_baseline + residual

        return predicted, uncertainty


class PINNLoss(nn.Module):
    """Custom loss function with physics-informed constraints.

    Total loss = L_data + λ₁·L_physics + λ₂·L_monotone + λ₃·L_boundary
    """

    def __init__(
        self,
        physics: PhysicsConstraints,
        lambda_physics: float = 0.5,
        lambda_monotone: float = 0.3,
        lambda_boundary: float = 0.2,
    ):
        super().__init__()
        self.physics = physics
        self.lambda_physics = lambda_physics
        self.lambda_monotone = lambda_monotone
        self.lambda_boundary = lambda_boundary
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        uncertainty: torch.Tensor,
        cycle_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss with component breakdown.

        Returns:
            total_loss: scalar tensor
            loss_components: dict with individual loss values for logging
        """
        # 1. Data loss: NLL with learned uncertainty (heteroscedastic)
        #    -log p(y|pred, sigma) ∝ log(sigma) + (y-pred)²/(2*sigma²)
        l_data = torch.mean(
            torch.log(uncertainty + 1e-6) + (target - pred) ** 2 / (2 * uncertainty ** 2 + 1e-6)
        )

        # 2. Physics loss: predicted capacity should be consistent with
        #    ECM-based degradation dynamics
        physics_cap = self.physics.expected_capacity_at_cycle(cycle_indices)
        # Ensure shape matches pred: (batch, 1)
        if physics_cap.dim() == 1:
            physics_cap = physics_cap.unsqueeze(-1)
        l_physics = self.mse(pred, physics_cap)

        # 3. Monotonicity loss: penalize capacity *increases* between consecutive
        #    cycles (capacity should generally decline)
        if len(pred) > 1:
            cap_diff = pred[1:] - pred[:-1]  # Should be ≤ 0
            # Only penalize increases, allow small recovery (tolerance 0.005 Ah)
            violations = torch.relu(cap_diff - 0.005)
            l_monotone = torch.mean(violations ** 2)
        else:
            l_monotone = torch.tensor(0.0, device=pred.device)

        # 4. Boundary loss: capacity within [0, rated_capacity * 1.05]
        cap_max = self.physics.rated_capacity * 1.05
        l_boundary = torch.mean(
            torch.relu(-pred) ** 2 + torch.relu(pred - cap_max) ** 2
        )

        total = l_data + self.lambda_physics * l_physics + \
                self.lambda_monotone * l_monotone + self.lambda_boundary * l_boundary

        components = {
            "data": l_data.item(),
            "physics": l_physics.item(),
            "monotone": l_monotone.item(),
            "boundary": l_boundary.item(),
            "total": total.item(),
        }

        return total, components


class PINNDataset(Dataset):
    """Dataset for PINN that includes cycle indices for physics constraints."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, cycle_indices: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
        self.cycles = torch.FloatTensor(cycle_indices).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx], self.cycles[idx]


def train_pinn(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_cycles: np.ndarray,
    physics: PhysicsConstraints,
    val_features: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    val_cycles: np.ndarray | None = None,
    hidden_dim: int = 64,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 25,
) -> tuple[PINNCapacityPredictor, dict]:
    """Train a PINN model.

    Returns the trained model and training history (including loss breakdown).
    """
    input_dim = train_features.shape[1]
    model = PINNCapacityPredictor(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    pinn_loss = PINNLoss(physics)

    train_ds = PINNDataset(train_features, train_targets, train_cycles)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_features is not None and val_targets is not None and val_cycles is not None:
        val_ds = PINNDataset(val_features, val_targets, val_cycles)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": [], "components": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_components = []

        for x_batch, y_batch, c_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            c_batch = c_batch.to(DEVICE)

            # Physics baseline
            physics_base = physics.expected_capacity_at_cycle(c_batch)

            optimizer.zero_grad()
            pred, uncertainty = model(x_batch, physics_base)
            loss, components = pinn_loss(pred, y_batch, uncertainty, c_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_components.append(components)

        scheduler.step()
        avg_train = np.mean(epoch_losses)
        history["train_loss"].append(avg_train)

        # Average component losses for this epoch
        avg_comp = {}
        for key in epoch_components[0]:
            avg_comp[key] = np.mean([c[key] for c in epoch_components])
        history["components"].append(avg_comp)

        # Validate
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_batch, y_batch, c_batch in val_loader:
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    c_batch = c_batch.to(DEVICE)
                    physics_base = physics.expected_capacity_at_cycle(c_batch)
                    pred, uncertainty = model(x_batch, physics_base)
                    loss, _ = pinn_loss(pred, y_batch, uncertainty, c_batch)
                    val_losses.append(loss.item())

            avg_val = np.mean(val_losses)
            history["val_loss"].append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info("PINN early stop at epoch %d", epoch + 1)
                    break
        else:
            if avg_train < best_val_loss:
                best_val_loss = avg_train
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    return model, history


def predict_pinn(
    model: PINNCapacityPredictor,
    features: np.ndarray,
    cycle_indices: np.ndarray,
    physics: PhysicsConstraints,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict capacity with uncertainty.

    Returns: (predictions, uncertainties)
    """
    model.eval()
    x = torch.FloatTensor(features).to(DEVICE)
    c = torch.FloatTensor(cycle_indices).unsqueeze(1).to(DEVICE)
    physics_base = physics.expected_capacity_at_cycle(c)

    with torch.no_grad():
        pred, uncertainty = model(x, physics_base)

    return pred.cpu().numpy().flatten(), uncertainty.cpu().numpy().flatten()


def predict_pinn_with_mc_dropout(
    model: PINNCapacityPredictor,
    features: np.ndarray,
    cycle_indices: np.ndarray,
    physics: PhysicsConstraints,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MC Dropout prediction with combined epistemic + aleatoric uncertainty.

    Returns: (mean_pred, lower_95, upper_95, aleatoric_uncertainty)
    """
    model.train()  # Enable dropout
    all_preds = []
    all_uncert = []

    x = torch.FloatTensor(features).to(DEVICE)
    c = torch.FloatTensor(cycle_indices).unsqueeze(1).to(DEVICE)
    physics_base = physics.expected_capacity_at_cycle(c)

    for _ in range(n_samples):
        with torch.no_grad():
            pred, uncert = model(x, physics_base)
            all_preds.append(pred.cpu().numpy().flatten())
            all_uncert.append(uncert.cpu().numpy().flatten())

    all_preds_arr = np.array(all_preds)
    all_uncert_arr = np.array(all_uncert)

    mean_pred = all_preds_arr.mean(axis=0)
    epistemic = all_preds_arr.std(axis=0)  # Model uncertainty
    aleatoric = all_uncert_arr.mean(axis=0)  # Data uncertainty

    # Combined uncertainty for CI
    total_std = np.sqrt(epistemic ** 2 + aleatoric ** 2)
    lower = mean_pred - 1.96 * total_std
    upper = mean_pred + 1.96 * total_std

    model.eval()
    return mean_pred, lower, upper, aleatoric
