# gnn_utils/inverse_design_utils.py

import torch
import numpy as np
from types import SimpleNamespace

def optimize_input_for_target(
    model,
    scaler_x,
    scaler_y,
    desired_error_orig,
    feature_dim,
    num_steps=800,
    lr=0.05,
    bounds_orig=None,
    integer_indices=None,
    device=None
):
    """
    MODE 2:
    Given desired_error, find X that MINIMIZES cpu_time
    while keeping predicted_error <= desired_error.
    """

    if device is None:
        device = next(model.parameters()).device

    # ---------------------------------------------------
    # 1. Convert desired error to normalized target
    # ---------------------------------------------------
    desired_target_orig = np.array([[desired_error_orig, 0.0]])  # cpu_time ignored
    y_scaled = scaler_y.transform(desired_target_orig)
    desired_error_scaled = torch.tensor(
        y_scaled[:, 0:1], dtype=torch.float32, device=device
    )  # only error dimension

    # ---------------------------------------------------
    # 2. Initialize X_var (normalized)
    # ---------------------------------------------------
    X_var = torch.zeros((1, feature_dim), dtype=torch.float32,
                        device=device, requires_grad=True)

    # ---------------------------------------------------
    # 3. Convert bounds to normalized space
    # ---------------------------------------------------
    lower_norm = upper_norm = None
    if bounds_orig is not None:
        mean = scaler_x.mean_
        scale = scaler_x.scale_

        lower_arr = np.full((feature_dim,), -1e9)
        upper_arr = np.full((feature_dim,),  1e9)

        for idx, (lo, hi) in bounds_orig.items():
            lower_arr[idx] = (lo - mean[idx]) / scale[idx]
            upper_arr[idx] = (hi - mean[idx]) / scale[idx]

        lower_norm = torch.tensor(lower_arr, dtype=torch.float32, device=device)
        upper_norm = torch.tensor(upper_arr, dtype=torch.float32, device=device)

    # ---------------------------------------------------
    # 4. Optimization loop (Mode 2 loss!)
    # ---------------------------------------------------
    optimizer = torch.optim.Adam([X_var], lr=lr)

    λ = 0.1   # small weight for cpu_time

    for step in range(num_steps):
        optimizer.zero_grad()

        dummy = SimpleNamespace()
        dummy.x = X_var
        dummy.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        pred_scaled = model(dummy)

        pred_error = pred_scaled[:, 0:1]   # shape (1,1)
        pred_cpu   = pred_scaled[:, 1:2]   # shape (1,1)

        # -----------------------------
        # MODE 2 LOSS:
        # enforce error <= desired_error
        # minimize cpu_time
        # -----------------------------
        error_excess = torch.relu(pred_error - desired_error_scaled)
        loss = error_excess**2 + λ * pred_cpu

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        # Clamp parameters into bounds
        if lower_norm is not None:
            with torch.no_grad():
                X_var.data = torch.max(torch.min(X_var.data, upper_norm), lower_norm)

        if step % 100 == 0:
            print(f"[inverse] step {step}, loss={loss.item():.6f}, "
                  f"error={pred_error.item():.4f}, cpu={pred_cpu.item():.4f}")

    # ---------------------------------------------------
    # 5. Convert optimized X back to original units
    # ---------------------------------------------------
    X_norm = X_var.detach().cpu().numpy()
    X_orig = scaler_x.inverse_transform(X_norm)

    # ---------------------------------------------------
    # 6. Round integer-only parameters
    # ---------------------------------------------------
    if integer_indices is not None:
        for idx in integer_indices:
            X_orig[0, idx] = np.round(X_orig[0, idx])

    # ---------------------------------------------------
    # 7. Predict y in original space
    # ---------------------------------------------------
    y_pred_scaled = pred_scaled.detach().cpu().numpy()
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

    return X_orig, X_norm, y_pred_orig
