import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup

from utils import load_data, create_dataloaders, create_weighted_criterion, MAX_LEN
from model import TransformerClassifier


def parse_args():
    """
    Parse command-line arguments required for training.

    Returns:
        argparse.Namespace: Parsed arguments including paths, model settings,
        training hyperparameters, and output configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="liar_dataset")
    parser.add_argument("--model_name", type=str, default="google/mobilebert-uncased")
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./checkpoint/best_model_all.pt")
    parser.add_argument("--plot_path", type=str, default="loss_curve.png")
    parser.add_argument("--skip", type=int, default=1)
    return parser.parse_args()


# ==========================
# Training
# ==========================
def train_epoch(loader, model, optimizer, scheduler, criterion, device):
    """
    Train the model for one epoch on the given dataloader.

    This function performs the standard training loop: moves data to the
    target device, computes predictions, evaluates loss, performs backpropagation,
    updates model parameters, and steps the learning rate scheduler.

    Args:
        loader (DataLoader): Training dataloader.
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        scheduler: Learning rate scheduler stepped after each batch.
        criterion: Loss function used for optimization.
        device (torch.device): Target device such as CPU or CUDA.

    Returns:
        float: Average training loss across all batches in the epoch.
    """
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def evaluate(loader, model, criterion, device):
    """
    Evaluate the model on a validation or test dataloader.

    The model is run in evaluation mode without gradient computation.
    This function collects the average loss, true labels, and predicted
    labels for downstream metric calculation.

    Args:
        loader (DataLoader): Validation or test dataloader.
        model (torch.nn.Module): Model to evaluate.
        criterion: Loss function used for evaluation.
        device (torch.device): Target device such as CPU or CUDA.

    Returns:
        tuple: A tuple containing:
            - float: Average loss over the dataset.
            - list: Ground-truth labels.
            - list: Predicted labels.
    """
    model.eval()
    total_loss = 0
    preds, true = [], []
    loop = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            true.extend(labels.cpu().numpy())
            loop.set_postfix(val_loss=loss.item())

    avg_loss = total_loss / len(loader)
    return avg_loss, true, preds


def main():
    """
    Run the complete training pipeline.

    This function parses arguments, loads and preprocesses the data,
    creates dataloaders, initializes the model and training components,
    performs epoch-wise training and validation, and saves the best
    model checkpoint based on validation performance.
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, tokenizer = load_data(
        args.data_dir,
        model_name=args.model_name,
        max_len=args.max_len,
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=args.batch_size
    )

    criterion = create_weighted_criterion(train_df, device)

    # ==========================
    # Setup
    # ==========================
    model = TransformerClassifier(args.model_name, dropout=args.dropout).to(device)

    criterion = criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # ==========================
    # Training Loop
    # ==========================
    best_val_f1 = -1
    patience = args.patience
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(train_loader, model, optimizer, scheduler, criterion, device)
        val_loss, y_true, y_pred = evaluate(val_loader, model, criterion, device)
        val_f1 = f1_score(y_true, y_pred, average="macro")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val F1:     {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), args.save_path)
            print("✅ Best model saved (by Val macro-F1)!")
        else:
            wait += 1
            if wait >= patience:
                print("⏹ Early stopping triggered.")
                break

    # ==========================
    # Plot Loss Curve
    # ==========================
    skip = args.skip  # number of initial epochs to skip

    n = min(len(train_losses), len(val_losses))  # safe if lengths differ
    if skip >= n:
        raise ValueError(f"skip={skip} is too large for {n} recorded epochs.")

    epochs = range(skip + 1, n + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses[skip:n], marker='o', label="Train Loss")
    plt.plot(epochs, val_losses[skip:n], marker='o', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss (from Epoch {skip + 1})")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.plot_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
