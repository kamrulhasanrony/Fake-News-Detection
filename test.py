import sys
import argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score

from utils import load_data, create_dataloaders, create_weighted_criterion, label_map, MAX_LEN
from model import TransformerClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="liar_dataset")
    parser.add_argument("--model_name", type=str, default="google/mobilebert-uncased")
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, default="./checkpoint/best_model_all.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def evaluate(loader, model, criterion, device):
    """
    Evaluate the trained model on the test dataloader.

    The function runs inference without gradient updates, computes the
    average loss, and collects true and predicted labels for reporting
    evaluation metrics.

    Args:
        loader (DataLoader): Test dataloader.
        model (torch.nn.Module): Trained model to evaluate.
        criterion: Loss function used during evaluation.
        device (torch.device): Target device such as CPU or CUDA.

    Returns:
        tuple: A tuple containing:
            - float: Average test loss.
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
    Run the complete evaluation pipeline.

    This function parses arguments, loads the test data and tokenizer,
    restores the trained model from checkpoint, performs evaluation,
    and prints classification metrics and related results.
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

    model = TransformerClassifier(args.model_name, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    report_file = open("report.txt", "w")
    sys.stdout = report_file
    if args.split == "val":
        # ==========================
        # Validation Evaluation
        # ==========================
        test_loss, y_true, y_pred = evaluate(val_loader, model, criterion, device)
        print(classification_report(y_true, y_pred, target_names=list(label_map.keys())))
    elif args.split == "test":
        # ==========================
        # Test Evaluation
        # ==========================
        test_loss, y_true, y_pred = evaluate(test_loader, model, criterion, device)
        # print(classification_report(y_true, y_pred, target_names=list(label_map.keys())))
        print(classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4, 5],  # force all classes
            target_names=list(label_map.keys())
        ))
    else:
        # ==========================
        # Evaluation
        # ==========================
        test_loss, y_true, y_pred = evaluate(train_loader, model, criterion, device)
        print(classification_report(y_true, y_pred, target_names=list(label_map.keys())))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(label_map.keys()),
                yticklabels=list(label_map.keys()))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    # plt.show()

    # ==========================
    # Additional Metrics
    # ==========================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("Unique y_true:", set(y_true))
    print("Unique y_pred:", set(y_pred))
    report_file.close()


if __name__ == "__main__":
    main()
