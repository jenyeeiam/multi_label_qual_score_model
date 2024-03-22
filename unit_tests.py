import unittest
import torch
from multi_label_class import MultiTaskBertModel
from data_preparation import prepare_dataset

class TestMultiTaskBertModel(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig()
        self.model = MultiTaskBertModel(self.config)

    def test_forward_with_labels(self):
        batch_size = 2
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, 128))
        attention_mask = torch.ones((batch_size, 128), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, 128), dtype=torch.long)
        evidence_labels = torch.randint(0, 4, (batch_size,))
        suggestion_labels = torch.randint(0, 2, (batch_size,))
        connection_labels = torch.randint(0, 2, (batch_size,))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            evidence_labels=evidence_labels,
            suggestion_labels=suggestion_labels,
            connection_labels=connection_labels,
        )

        self.assertIn("loss", outputs)
        self.assertTrue(outputs["loss"].requires_grad)

    def test_compute_metrics(self):
        batch_size = 2
        evidence_logits = torch.randn(batch_size, 4)
        suggestion_logits = torch.randn(batch_size, 2)
        connection_logits = torch.randn(batch_size, 2)
        evidence_labels = torch.randint(0, 4, (batch_size,))
        suggestion_labels = torch.randint(0, 2, (batch_size,))
        connection_labels = torch.randint(0, 2, (batch_size,))

        outputs = {
            "evidence": evidence_logits,
            "suggestion": suggestion_logits,
            "connection": connection_logits,
        }

        metrics = self.model.compute_metrics(
            outputs, evidence_labels, suggestion_labels, connection_labels
        )

        self.assertIn("evidence_acc", metrics)
        self.assertIn("evidence_f1", metrics)
        self.assertIn("evidence_auc", metrics)
        self.assertIn("suggestion_acc", metrics)
        self.assertIn("suggestion_f1", metrics)
        self.assertIn("suggestion_auc", metrics)
        self.assertIn("connection_acc", metrics)
        self.assertIn("connection_f1", metrics)
        self.assertIn("connection_auc", metrics)

        # Check that the AUC values are within the valid range (0.0 to 1.0)
        self.assertTrue(0.0 <= metrics["evidence_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["suggestion_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["connection_auc"] <= 1.0)

class TestDataPreparation(unittest.TestCase):
    def test_prepare_dataset(self):
        train_dataloader, val_dataloader, test_dataloader = prepare_dataset('./tests/test_data.json')
        
        # Check that the dataloaders are not empty
        self.assertGreater(len(train_dataloader), 0)
        self.assertGreater(len(val_dataloader), 0)
        self.assertGreater(len(test_dataloader), 0)
        
        # Check the shape of the tensors in the batch
        for batch in train_dataloader:
            self.assertEqual(batch["input_ids"].shape[0], 16)  # Batch size
            self.assertEqual(batch["input_ids"].shape[1], 512)  # Max sequence length
            self.assertEqual(batch["evidence"].shape[0], 16)
            self.assertEqual(batch["suggestion"].shape[0], 16)
            self.assertEqual(batch["connection"].shape[0], 16)
            break  # Check only one batch

if __name__ == "__main__":
    unittest.main()
