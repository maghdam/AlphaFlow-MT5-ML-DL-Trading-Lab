import unittest
import pandas as pd
import numpy as np
from models.model_training import walk_forward_splits

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Create a simple dummy sequence (e.g., numbers 0 to 9)
        self.X = pd.DataFrame({'feature': np.arange(10)})
        self.y = pd.Series(np.arange(10))

    def test_walk_forward_splits_basic(self):
        """Test basic chronological splitting with default n_splits=3 on n=10."""
        # For n=10, n_splits=3, fold_size = 10 // (3 + 1) = 2
        # Fold 1: Train [0:2], Test [2:4]
        # Fold 2: Train [0:4], Test [4:6]
        # Fold 3: Train [0:6], Test [6:8]
        folds = walk_forward_splits(self.X, self.y, n_splits=3)

        self.assertEqual(len(folds), 3)

        # Fold 1
        X_train_1, y_train_1, X_test_1, y_test_1 = folds[0]
        self.assertEqual(len(X_train_1), 2)
        self.assertEqual(len(X_test_1), 2)
        self.assertTrue(np.array_equal(X_train_1['feature'].values, [0, 1]))
        self.assertTrue(np.array_equal(X_test_1['feature'].values, [2, 3]))

        # Fold 2
        X_train_2, y_train_2, X_test_2, y_test_2 = folds[1]
        self.assertEqual(len(X_train_2), 4)
        self.assertEqual(len(X_test_2), 2)
        self.assertTrue(np.array_equal(X_train_2['feature'].values, [0, 1, 2, 3]))
        self.assertTrue(np.array_equal(X_test_2['feature'].values, [4, 5]))

        # Fold 3
        X_train_3, y_train_3, X_test_3, y_test_3 = folds[2]
        self.assertEqual(len(X_train_3), 6)
        self.assertEqual(len(X_test_3), 2)
        self.assertTrue(np.array_equal(X_train_3['feature'].values, [0, 1, 2, 3, 4, 5]))
        self.assertTrue(np.array_equal(X_test_3['feature'].values, [6, 7]))

    def test_walk_forward_splits_varying_n_splits(self):
        """Test varying number of splits."""
        # For n=10, n_splits=4, fold_size = 10 // (4 + 1) = 2
        folds = walk_forward_splits(self.X, self.y, n_splits=4)
        self.assertEqual(len(folds), 4)

        # Last fold: Train [0:8], Test [8:10]
        X_train_last, _, X_test_last, _ = folds[-1]
        self.assertEqual(len(X_train_last), 8)
        self.assertEqual(len(X_test_last), 2)
        self.assertEqual(X_test_last['feature'].iloc[-1], 9)

    def test_walk_forward_splits_remainder(self):
        """Test dataset size that doesn't divide perfectly."""
        # n=11, n_splits=3, fold_size = 11 // (3 + 1) = 2
        # Fold 1: Train [0:2], Test [2:4]
        # Fold 2: Train [0:4], Test [4:6]
        # Fold 3: Train [0:6], Test [6:8]
        X_11 = pd.DataFrame({'feature': np.arange(11)})
        y_11 = pd.Series(np.arange(11))

        folds = walk_forward_splits(X_11, y_11, n_splits=3)
        self.assertEqual(len(folds), 3)

        # Check last fold bounds
        X_train_last, _, X_test_last, _ = folds[-1]
        self.assertEqual(len(X_train_last), 6)
        self.assertEqual(len(X_test_last), 2)
        self.assertTrue(np.array_equal(X_test_last['feature'].values, [6, 7]))

    def test_walk_forward_splits_end_bounds(self):
        """Test case where end_test is capped at n."""
        # n=5, n_splits=2, fold_size = 5 // 3 = 1
        # Fold 1: Train [0:1], Test [1:2]
        # Fold 2: Train [0:2], Test [2:3]
        X_5 = pd.DataFrame({'feature': np.arange(5)})
        y_5 = pd.Series(np.arange(5))
        folds = walk_forward_splits(X_5, y_5, n_splits=2)
        self.assertEqual(len(folds), 2)

        X_train_last, _, X_test_last, _ = folds[-1]
        self.assertEqual(len(X_train_last), 2)
        self.assertEqual(len(X_test_last), 1)
        self.assertTrue(np.array_equal(X_test_last['feature'].values, [2]))


if __name__ == '__main__':
    unittest.main()
