import argparse
import os
import torch
import numpy as np
import sys
import pandas as pd
sys.path.append("../")

from gnn_utils.gnn_model_utils import (
    build_unified_err_graph,
    load_graph,
    mask_data,
    GCN
)

# ------------------------------------
# Train function
# ------------------------------------
def train_gnn(csv_path, feature_cols, graph_path, label, epochs=2000, lr=0.01, split=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------------------------
    # Step 1: Build graph if not already saved
    # --------------------------------------------------
    if not os.path.exists(graph_path):
        print("Graph not found — building new graph...")
        build_unified_err_graph(csv_path, feature_cols, out_path=graph_path)

    # --------------------------------------------------
    # Step 2: Load graph & scalers
    # --------------------------------------------------
    data, scaler_x, scaler_y = load_graph(graph_path)
    data = mask_data(data, split=split)

    # --------------------------------------------------
    # Step 3: Build model
    # --------------------------------------------------
    model = GCN(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        out_channels=data.y.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # --------------------------------------------------
    # Step 4: Training loop
    # --------------------------------------------------
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    # --------------------------------------------------
    # Step 5: Make predictions (un-normalized)
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        pred_scaled = model(data).cpu().numpy()
        pred_orig = scaler_y.inverse_transform(pred_scaled)

    # Attach to dataframe
    df_out = data.raw_df.copy()
    df_out[["pred_error", "pred_cpu_time"]] = pred_orig

    out_csv = os.path.join(os.path.dirname(graph_path), "gnn_"+label+"_predictions.csv")
    df_out.to_csv(out_csv, index=False)

    model_path = os.path.join(os.path.dirname(graph_path), f"gnn_{label}_model.pt")

    torch.save({
        "model_state": model.state_dict(),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "graph_path": graph_path
    }, model_path)

    print(f"Saved predictions to {out_csv}")
    print(f"Saved trained model to {model_path}")
    return model, data, scaler_x, scaler_y


# =====================================
# Command-line interface
# =====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--feature_cols", nargs="+", required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--split", type=float, default=0.8)

    args = parser.parse_args()

    train_gnn(
        csv_path=args.csv_path,
        feature_cols=args.feature_cols,
        graph_path=args.graph_path,
        label=args.label,
        epochs=args.epochs,
        lr=args.lr,
        split=args.split
    )

#Below is how to run this code in terminal

# python train_gnn.py \
#     --csv_path Data/class_dataset.csv \
#     --feature_cols tol_background tol_thermo \
#     --graph_path Data/class_graph.pt
