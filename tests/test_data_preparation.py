# test_data_preparation.py
import unittest
from data_preparation import prepare_dataset


class TestDataPreparation(unittest.TestCase):
    def test_prepare_dataset(self):
        train_dataloader, val_dataloader, test_dataloader = prepare_dataset(
            "./data/test_data.json"
        )

        # Check that the dataloaders are not empty
        self.assertGreater(len(train_dataloader), 0)
        self.assertGreater(len(val_dataloader), 0)
        self.assertGreater(len(test_dataloader), 0)

        # Check that the batch size is correct
        self.assertEqual(train_dataloader.batch_size, 16)
        self.assertEqual(val_dataloader.batch_size, 16)
        self.assertEqual(test_dataloader.batch_size, 16)

        # Check that the data is on the correct device
        # self.assertEqual(next(iter(train_dataloader))["input_ids"].device, "cpu")


if __name__ == "__main__":
    unittest.main()
