import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils.generate_data import SurfaceWaveDataset
from models.struct2disp_transformer import Struct2DispTransformer
plt.switch_backend("Agg")  # Use non-interactive backend for plotting

def train_struct2disp_transformer(
    loader,
    n_epoch=200,
    lr=2e-4,
    weight_decay=1e-4,
    ckpt_path="ckpt/struct2disp_transformer.v1.1.pt",
    fig_dir="tfig",
    device=None,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # -------- device --------
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print("Device:", device)

    # -------- peek one batch --------
    model_batch, disp_batch, mask_batch = next(iter(loader))
    B, C_model, H = model_batch.shape        # [B,4,H]
    _, C_disp, T = disp_batch.shape          # [B,3,T]
    print(f"H={H}, T={T}")

    assert T == 59, "This script assumes T=59"

    # -------- model --------
    model = Struct2DispTransformer(
        H=H,
        T=T,
        C_in=4,
        d_model=512,
        nhead=8,
        num_enc_layers=8,
        num_dec_layers=8,
        dim_ff=1024,
        dropout=0.1,
        use_period_values=True,
        period_minmax=(disp_batch[:, 0].min().item(),
                       disp_batch[:, 0].max().item()),
    ).to(device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded checkpoint:", ckpt_path)

    # -------- loss & optimizer --------
    # 先只训 mu，用 SmoothL1（Huber）
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    global_step = 0

    # -------- training loop --------
    for epoch in range(n_epoch):
        model.train()
        for model_batch, disp_batch, _ in loader:
            model_batch = model_batch.to(device)      # [B,4,H]
            disp_batch = disp_batch.to(device)        # [B,3,T]

            x = model_batch
            periods = disp_batch[:, 0, :]             # [B,T]
            y_true = disp_batch[:, 1:, :]             # [B,2,T]

            mu_pred, _ = model(x, periods=periods)

            loss = loss_fn(mu_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            global_step += 1

            # ---- logging & plotting ----
            if global_step % 200 == 0:
                torch.save(model.state_dict(), ckpt_path)
                print(f"[Epoch {epoch:03d} | Step {global_step:06d}] "
                      f"loss={loss.item():.6f}")

                with torch.no_grad():
                    T_grid = periods[0].cpu().numpy()
                    cR_true = y_true[0, 0].cpu().numpy()
                    cL_true = y_true[0, 1].cpu().numpy()
                    cR_pred = mu_pred[0, 0].cpu().numpy()
                    cL_pred = mu_pred[0, 1].cpu().numpy()

                plt.figure(figsize=(6, 5))
                plt.plot(T_grid, cR_true, "b-", label="Rayleigh true")
                plt.plot(T_grid, cR_pred, "b--", label="Rayleigh pred")
                plt.plot(T_grid, cL_true, "r-", label="Love true")
                plt.plot(T_grid, cL_pred, "r--", label="Love pred")
                plt.xlabel("Period (s)")
                plt.ylabel("Phase velocity (km/s)")
                plt.title(f"Step={global_step}, loss={loss.item():.3f}")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.savefig(
                    os.path.join(fig_dir, "disp_fit_step.pdf"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

        torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch} finished, checkpoint saved.")

    print("Training finished.")
    return model


# =========================
# Example entry
# =========================
if __name__ == "__main__":
    ds = SurfaceWaveDataset(
        n_samples=100_000,
        z_max_km=150.0,
        z_max_num=256,
        dz_km=0.5,
        seed=2026,
    )

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        num_workers=36,
        pin_memory=True,
    )

    train_struct2disp_transformer(loader)
