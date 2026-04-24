import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ==========================
# Load LIAR dataset
# ==========================
# cols = list(range(14))  # LIAR has 14 columns

columns = [
    "id",  # Column 1
    "label",  # Column 2
    "statement",  # Column 3
    "subject",  # Column 4
    "speaker",  # Column 5
    "speaker_job_title",  # Column 6
    "state_info",  # Column 7
    "party_affiliation",  # Column 8
    "barely_true_counts",  # Column 9
    "false_counts",  # Column 10
    "half_true_counts",  # Column 11
    "mostly_true_counts",  # Column 12
    "pants_on_fire_counts",  # Column 13
    "context"  # Column 14
]

label_map = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

MAX_LEN = 192
text_cols = [
    "statement", "subject", "speaker",
    "speaker_job_title", "state_info",
    "party_affiliation", "context"
]
num_cols = [
    "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts"
]


def build_text(row):
    """
    Construct a single input text string from multiple fields in a row.

    The function combines relevant metadata and text fields into one
    formatted string so the transformer model can consume all useful
    information as a single sequence.

    Args:
        row (pd.Series): A single dataframe row containing the source fields.

    Returns:
        str: Combined text representation for model input.
    """
    return (
        f"Statement: {row['statement']} "
        f"Subject: {row['subject']} "
        f"Speaker: {row['speaker']} "
        f"Job: {row['speaker_job_title']} "
        f"State: {row['state_info']} "
        f"Party: {row['party_affiliation']} "
        f"Context: {row['context']} "
        f"History: barely_true={row['barely_true_counts']}, "
        f"false={row['false_counts']}, "
        f"half_true={row['half_true_counts']}, "
        f"mostly_true={row['mostly_true_counts']}, "
        f"pants_fire={row['pants_on_fire_counts']}"
    )


# ==========================
# Custom Dataset
# ==========================
class TextDataset(Dataset):
    """
    PyTorch dataset for tokenizing text samples and returning model-ready inputs.

    Each item includes tokenized input ids, attention mask, and the
    corresponding label, making it suitable for transformer-based
    text classification training and evaluation.
    """
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        """
        Initialize the dataset with dataframe records and tokenizer settings.

        Args:
            df (pd.DataFrame): Dataframe containing text inputs and labels.
            tokenizer: Hugging Face tokenizer used to encode text.
            max_len (int, optional): Maximum token sequence length.
                Defaults to MAX_LEN.
        """
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of examples available in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieve and tokenize a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing tokenized tensors such as
            input_ids, attention_mask, and the target label.
        """
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(data_dir, model_name="google/mobilebert-uncased", max_len=MAX_LEN):
    """
    Load dataset files, preprocess text fields, and initialize the tokenizer.

    This function reads the train, validation, and test splits, applies
    required text preparation steps, and creates the tokenizer associated
    with the selected pretrained transformer model.

    Args:
        data_dir (str): Directory containing the dataset split files.
        model_name (str, optional): Name or path of the pretrained tokenizer.
            Defaults to "google/mobilebert-uncased".
        max_len (int, optional): Maximum sequence length used later for
            tokenization. Defaults to MAX_LEN.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Training dataframe.
            - pd.DataFrame: Validation dataframe.
            - pd.DataFrame: Test dataframe.
            - tokenizer: Initialized tokenizer for the chosen model.
    """
    train_df = pd.read_csv(f"{data_dir}/train.tsv", sep="\t", header=None, names=columns)
    val_df = pd.read_csv(f"{data_dir}/valid.tsv", sep="\t", header=None, names=columns)
    test_df = pd.read_csv(f"{data_dir}/test.tsv", sep="\t", header=None, names=columns)

    train_df = train_df[
        ['statement', 'subject', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation', 'context',
         'barely_true_counts', 'false_counts',
         'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'label']].copy()
    val_df = val_df[
        ['statement', 'subject', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation', 'context',
         'barely_true_counts', 'false_counts',
         'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'label']].copy()
    test_df = test_df[
        ['statement', 'subject', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation', 'context',
         'barely_true_counts', 'false_counts',
         'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'label']].copy()

    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].str.strip().str.lower().map(label_map)

    # Fill text fields
    for col in text_cols:
        train_df[col] = train_df[col].fillna("unknown")
        val_df[col] = val_df[col].fillna("unknown")
        test_df[col] = test_df[col].fillna("unknown")

    # Fill numeric fields
    for col in num_cols:
        train_df[col] = train_df[col].fillna(0)
        val_df[col] = val_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)

    train_df["text"] = train_df.apply(build_text, axis=1)
    val_df["text"] = val_df.apply(build_text, axis=1)
    test_df["text"] = test_df.apply(build_text, axis=1)

    max_len_words = train_df["text"].apply(lambda x: len(x.split())).max()
    print("Max length (words):", max_len_words)

    print(" Train:", len(train_df), "      Validation:", len(val_df), "      Test:", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df = train_df[train_df["text"].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=True, truncation=False)) <= max_len)].reset_index(
        drop=True)
    test_df = test_df[test_df["text"].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=True, truncation=False)) <= max_len)].reset_index(
        drop=True)
    val_df = val_df[val_df["text"].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=True, truncation=False)) <= max_len)].reset_index(
        drop=True)

    print(" Train:", len(train_df), "      Validation:", len(val_df), "      Test:", len(test_df))

    max_len_words = train_df["text"].apply(lambda x: len(x.split())).max()
    print("Max length (words):", max_len_words)

    return train_df, val_df, test_df, tokenizer


def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=64):
    """
    Create PyTorch dataloaders for training, validation, and testing.

    This function wraps each dataframe split in the custom dataset class
    and returns dataloaders with the configured batch size for model
    training and evaluation.

    Args:
        train_df (pd.DataFrame): Training split dataframe.
        val_df (pd.DataFrame): Validation split dataframe.
        test_df (pd.DataFrame): Test split dataframe.
        tokenizer: Tokenizer used to encode the text samples.
        batch_size (int, optional): Number of samples per batch.
            Defaults to 64.

    Returns:
        tuple: A tuple containing train, validation, and test dataloaders.
    """
    train_loader = DataLoader(TextDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df, tokenizer), batch_size=batch_size)
    test_loader = DataLoader(TextDataset(test_df, tokenizer), batch_size=batch_size)
    return train_loader, val_loader, test_loader


def create_weighted_criterion(train_df, device):
    """
    Create a class-weighted cross-entropy loss function.

    Class weights are computed from the training label distribution to
    reduce the effect of class imbalance during optimization.

    Args:
        train_df (pd.DataFrame): Training dataframe containing class labels.
        device (torch.device): Device on which the loss weights should be placed.

    Returns:
        torch.nn.Module: Cross-entropy loss configured with class weights.
    """
    classes = np.array(sorted(train_df["label"].unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"]
    )

    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion
