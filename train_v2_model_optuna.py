import argparse
import csv
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from fvcore.nn.precise_bn import update_bn_stats
from model_v2_compatible import SeqNN
import schedulefree
import optuna
import pandas as pd


# =====================
# Dataset Definition
# =====================
class HiCDataset(Dataset):
    def __init__(self, data_files):
        self.data = []  # To store all sequences and HiC vectors
        
        # Load and process the data files
        for file in data_files:
            print("Loading file:", file)
            file_data = torch.load(file, weights_only=True)
            
            for data in file_data:
                ohe_sequence, hic_vector = data

                # Process the OHE sequence
                ohe_sequence = ohe_sequence.squeeze(0)  # Remove singleton dimension
                
                # Ensure the sequence has the correct shape
                assert ohe_sequence.shape[0] == 4, f"Expected 4 channels, but got {ohe_sequence.shape[0]}"
                assert len(ohe_sequence.shape) == 2, f"Expected 2D shape for sequence, but got {ohe_sequence.shape}"
                
                # Add processed pair to the data list
                self.data.append((ohe_sequence, hic_vector))
        
        print(f"Total sequences loaded: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the preprocessed (ohe_sequence, hic_vector) pair from memory
        ohe_sequence, hic_vector = self.data[idx]
        return ohe_sequence, hic_vector


# =====================
# Utility Functions
# =====================
def data_loader_for_precise_bn(loader, device):
    for data, _ in loader:
        yield data.to(device)


# =====================
# Train + Validation Loop
# =====================
def train_epoch(args, model, device, train_loader, val_loader, optimizer, epoch, weight_clip_value):
    model.train()
    optimizer.train()
    train_loss, num_batches = 0, 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        valid_mask = ~torch.isnan(target)
        loss = F.mse_loss(output[valid_mask], target[valid_mask])
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-weight_clip_value, weight_clip_value)

        train_loss += loss.item()
        num_batches += 1

    train_loss /= num_batches

    # update batchnorm
    update_bn_stats(model, data_loader_for_precise_bn(train_loader, device), num_iters=min(len(train_loader), 200))

    # validation
    model.eval()
    val_loss, num_batches = 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_mask = ~torch.isnan(target)
            if valid_mask.any():
                loss = F.mse_loss(output[valid_mask], target[valid_mask])
                val_loss += loss.item()
                num_batches += 1

    val_loss /= num_batches
    return train_loss, val_loss


# =====================
# Objective for Optuna
# =====================
def objective(trial, args, train_loader, val_loader, device):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_clip = trial.suggest_float('weight_clip', 5.0, 25.0)
    l2_scale = trial.suggest_loguniform('l2_scale', 1e-6, 1e-3)
    early_stop = trial.suggest_int('early_stop_patience', 5, 70)
    
    # Create fresh model
    model = SeqNN()
    model.to(device)
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=l2_scale)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\nStarting trial {trial.number} with lr={lr:.5g}, weight_clip={weight_clip}, l2_scale={l2_scale}, early_stop={early_stop}")
    
    for epoch in range(1, 15):  # Shorter run for Optuna search
        train_loss, val_loss = train_epoch(args, model, device, train_loader, val_loader, optimizer, epoch, weight_clip)
        
        if epoch % 3 == 0 or epoch == 1:
            print(f"Trial {trial.number} | Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        trial.report(val_loss, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= early_stop:
            print(f"Trial {trial.number} early stopping at epoch {epoch}")
            break
        
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.TrialPruned()
    
    print(f"Trial {trial.number} completed. Best Val Loss: {best_val_loss:.4f}\n")
    return best_val_loss


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--test_fold", type=str, required=True)
    parser.add_argument("--val_fold", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--organism", type=str, required=True)
    parser.add_argument("--data_split", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    # --- Setup ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    
    # --- Load data ---
    all_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".pt")]
    test_files = [f for f in all_files if args.test_fold in f]
    val_files = [f for f in all_files if args.val_fold in f]
    train_files = [f for f in all_files if args.test_fold not in f and args.val_fold not in f]
    
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}, Test files: {len(test_files)}")
    print("Loading data")
    
    train_dataset = HiCDataset(train_files)
    val_dataset = HiCDataset(val_files)
    
    # differece: Akita.v2 has a parameter shuffle_buffer=128
    # Here, the entire train set is shuffled randomly at the beginning of each epoch, so it's more randomized than using shuffle_buffer
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("len train_loader", len(train_loader))
    print("len valid_loader", len(valid_loader))
    
    # Run Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args, train_loader, valid_loader, device), n_trials=args.n_trials)

    df = study.trials_dataframe()
    df.to_csv(f"/scratch1/smaruj/Akita_pytorch_models/finetuned/{args.organism}_models/{args.data_name}/losses/optuna_{args.data_name}_model{args.data_split}_from_scratch_all_trials.csv", index=False)
    print(f"Saved all trial results to optuna_{args.data_name}_all_trials.csv")
    
    best_trial = study.best_trial
    best_info = {
        "trial_number": best_trial.number,
        "validation_loss": best_trial.value,
        **best_trial.params
    }

    # Save as CSV
    pd.DataFrame([best_info]).to_csv(f"/scratch1/smaruj/Akita_pytorch_models/finetuned/{args.organism}_models/{args.data_name}/losses/optuna_{args.data_name}_model{args.data_split}_from_scratch_best_trial.csv", index=False)
    print(f"Saved best trial info to optuna_{args.data_name}_best_trial.csv")

if __name__ == '__main__':
    main()

