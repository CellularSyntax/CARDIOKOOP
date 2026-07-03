import torch, sys
sys.path.insert(0, "src")

for name, path in [
    ("GRU",    "results/gru/csv_data_500_12sigs_20250625_170206_model.ckpt"),
    ("LSTM",   "results/lstm/csv_data_500_12sigs_20250625_184322_model.ckpt"),
    ("BiLSTM", "results/bilstm/csv_data_500_12sigs_20250625_204256_model.ckpt"),
]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    p = ckpt["params"]
    print(f"\n{name}:", {k: v for k, v in p.items()})
