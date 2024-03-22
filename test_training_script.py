# test_training_script.py
import unittest
import torch
from multi_label_class import MultiTaskBertModel
from training_script import initialize_model, train_model, evaluate_model

class TestTrainingScript(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = initialize_model(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=10, num_training_steps=100
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randint(0, 100, (16, 128)),
                torch.ones(16, 128, dtype=torch.long),
                torch.randint(0, 4, (16,)),
                torch.randint(0, 2, (16,)),
                torch.randint(0, 2, (16,))
            ),
            batch_size=16, shuffle=True
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randint(0, 100, (8, 128)),
                torch.ones(8, 128, dtype=torch.long),
                torch.randint(0, 4, (8,)),
                torch.randint(0, 2, (8,)),
                torch.randint(0, 2, (8,))
            ),
            batch_size=8
        )

    def test_train_model(self):
        trained_model = train_model(
            self.model, self.train_dataloader, self.val_dataloader, 2, self.optimizer, self.scheduler
        )
        self.assertIsInstance(trained_model, MultiTaskBertModel)

    def test_evaluate_model(self):
        test_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randint(0, 100, (8, 128)),
                torch.ones(8, 128, dtype=torch.long),
                torch.randint(0, 4, (8,)),
                torch.randint(0, 2, (8,)),
                torch.randint(0, 2, (8,))
            ),
            batch_size=8
        )
        test_loss, metrics = evaluate_model(self.model, test_dataloader)
        self.assertIsInstance(test_loss, float)
        for metric, value in metrics.items():
            self.assertIsInstance(value, float)
            self.assertTrue(0.0 <= value <= 1.0)

if __name__ == "__main__":
    unittest.main()
