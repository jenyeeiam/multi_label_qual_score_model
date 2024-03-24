from sklearn.preprocessing import label_binarize
import torch
import numpy as np

from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score


class MultiTaskBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.evidence_classifier = nn.Linear(
            config.hidden_size, 4
        )  # 4 possible outcomes
        self.suggestion_classifier = nn.Linear(
            config.hidden_size, 1
        )  # Output dimension for binary classification
        self.connection_classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        evidence_labels=None,
        suggestion_labels=None,
        connection_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output

        evidence_logits = self.evidence_classifier(pooled_output)
        suggestion_logits = self.suggestion_classifier(pooled_output)
        connection_logits = self.connection_classifier(pooled_output)

        outputs = {
            "evidence": evidence_logits,
            "suggestion": suggestion_logits,
            "connection": connection_logits,
        }

        total_loss = 0.0
        if evidence_labels is not None:
            evidence_loss = F.cross_entropy(
                evidence_logits, evidence_labels.squeeze(-1)
            )
            total_loss += evidence_loss
        if suggestion_labels is not None:
            suggestion_labels = suggestion_labels.float().unsqueeze(
                1
            )  # Add singleton dimension for logits
            print(f"suggestion_logits.shape = {suggestion_logits.shape}")
            print(f"suggestion_labels.shape = {suggestion_labels.shape}")
            suggestion_loss = F.binary_cross_entropy_with_logits(
                suggestion_logits, suggestion_labels
            )
            total_loss += suggestion_loss
        if connection_labels is not None:
            connection_labels = connection_labels.float().unsqueeze(1)
            connection_loss = F.binary_cross_entropy_with_logits(
                connection_logits, connection_labels
            )
            total_loss += connection_loss

        outputs["loss"] = total_loss
        return outputs

    def compute_metrics(
        self, outputs, evidence_labels, suggestion_labels, connection_labels
    ):
        # Extract logits from the model's output
        evidence_logits = outputs["evidence"]
        suggestion_logits = outputs["suggestion"]
        connection_logits = outputs["connection"]

        # Convert logits to probabilities
        # For evidence, use softmax since it's multi-class
        evidence_probs = F.softmax(evidence_logits, dim=1).cpu().detach().numpy()
        # For suggestion and connection, use sigmoid since they're binary
        suggestion_probs = (
            torch.sigmoid(suggestion_logits).squeeze(-1).cpu().detach().numpy()
        )
        connection_probs = (
            torch.sigmoid(connection_logits).squeeze(-1).cpu().detach().numpy()
        )

        # Convert evidence labels to one-hot encoding for AUC calculation
        evidence_labels_np = evidence_labels.cpu().numpy()
        evidence_labels_one_hot = label_binarize(
            evidence_labels_np, classes=np.arange(evidence_logits.size(1))
        )

        # Ensure evidence_probs is softmax probabilities if it's not already
        # evidence_probs = F.softmax(evidence_logits, dim=1).cpu().detach().numpy()

        # Calculate predictions for accuracy and F1 score
        # For evidence, pick the class with the highest probability
        evidence_preds = torch.argmax(evidence_logits, dim=1).cpu().numpy()
        # For suggestion and connection, threshold probabilities at 0.5
        suggestion_preds = (suggestion_probs > 0.5).astype(int)
        connection_preds = (connection_probs > 0.5).astype(int)

        # Calculate accuracy and F1 scores
        evidence_acc = accuracy_score(evidence_labels_np, evidence_preds)
        evidence_f1 = f1_score(evidence_labels_np, evidence_preds, average="macro")

        suggestion_acc = accuracy_score(
            suggestion_labels.cpu().numpy(), suggestion_preds
        )
        suggestion_f1 = f1_score(suggestion_labels.cpu().numpy(), suggestion_preds)

        connection_acc = accuracy_score(
            connection_labels.cpu().numpy(), connection_preds
        )
        connection_f1 = f1_score(connection_labels.cpu().numpy(), connection_preds)

        # Calculate AUC scores
        evidence_auc = roc_auc_score(
            evidence_labels_one_hot, evidence_probs, multi_class="ovr", average="macro"
        )
        suggestion_auc = roc_auc_score(
            suggestion_labels.cpu().numpy(), suggestion_probs
        )
        connection_auc = roc_auc_score(
            connection_labels.cpu().numpy(), connection_probs
        )

        # Compile metrics into a dictionary
        metrics = {
            "evidence_acc": evidence_acc,
            "evidence_f1": evidence_f1,
            "evidence_auc": evidence_auc,
            "suggestion_acc": suggestion_acc,
            "suggestion_f1": suggestion_f1,
            "suggestion_auc": suggestion_auc,
            "connection_acc": connection_acc,
            "connection_f1": connection_f1,
            "connection_auc": connection_auc,
        }

        return metrics
