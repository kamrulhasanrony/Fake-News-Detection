import torch
import torch.nn as nn
from transformers import AutoModel


# ==========================
# Model
# ==========================
class TransformerClassifier(nn.Module):
    """
    Multi-class text classification model built on top of a pretrained
    transformer encoder.

    The model uses the pooled output from the transformer backbone,
    applies dropout for regularization, and passes the result through
    a linear layer to produce class logits.
    """

    def __init__(self, model_name, num_classes=6, dropout=0.5):
        """
        Initialize the transformer-based classification model.

        Args:
            model_name (str): Name or path of the pretrained transformer model.
            num_classes (int, optional): Number of output classes. Defaults to 6.
            dropout (float, optional): Dropout rate applied before the final
                classification layer. Defaults to 0.5.
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)  # ✅ correct
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)

        # for param in self.encoder.parameters():
        # param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Token ids of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask indicating valid tokens.

        Returns:
            torch.Tensor: Raw logits for each class with shape
            (batch_size, num_classes).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        x = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
