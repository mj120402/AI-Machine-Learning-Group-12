import unittest

from cifar10_research_lab.data import choose_balanced_indices, limit_dataset


class DataUtilityTests(unittest.TestCase):
    def test_choose_balanced_indices_selects_the_same_number_from_each_class(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2]

        indices = choose_balanced_indices(labels, samples_per_class=2, seed=7)

        self.assertEqual(len(indices), 6)
        self.assertEqual(sorted(labels[index] for index in indices), [0, 0, 1, 1, 2, 2])

    def test_limit_dataset_keeps_labels_and_items_aligned(self):
        items = ["a", "b", "c", "d"]
        labels = [3, 2, 1, 0]

        limited_items, limited_labels = limit_dataset(items, labels, max_items=2)

        self.assertEqual(limited_items, ["a", "b"])
        self.assertEqual(limited_labels, [3, 2])

    def test_limit_dataset_returns_original_data_when_limit_is_none(self):
        items = ["a", "b"]
        labels = [1, 0]

        limited_items, limited_labels = limit_dataset(items, labels, max_items=None)

        self.assertIs(limited_items, items)
        self.assertIs(limited_labels, labels)


if __name__ == "__main__":
    unittest.main()
