import unittest
from dns_attacks_detection import *
from sklearn.datasets import make_classification


class TestPlotting(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=10, n_features=2, n_redundant=0, n_informative=2, random_state=13, n_clusters_per_class=1)
        self.figure, self.axes = plt.subplots()

    def test_plot_dataset(self):
        plot_dataset(self.X, self.y, "first_feature", "second_feature")
        # Test the plot legend
        self.assertEqual(2, len(self.axes.get_legend().get_texts()))
        no_attack_legend = self.axes.get_legend().get_texts()[0].get_text()
        attack_legend = self.axes.get_legend().get_texts()[1].get_text()
        self.assertEqual('No attack', no_attack_legend)
        self.assertEqual('Attack', attack_legend)
        # Test the plot labels
        label_x = self.axes.get_xlabel()
        label_y = self.axes.get_ylabel()
        self.assertEqual('first_feature', label_x)
        self.assertEqual('second_feature', label_y)
        # Test grid visibility
        # Although it is not the best practice to access a private variable,
        # it seems the only way to check the axis existance
        self.assertTrue(self.axes.xaxis._gridOnMinor and self.axes.xaxis._gridOnMajor)
        self.assertTrue(self.axes.yaxis._gridOnMinor and self.axes.yaxis._gridOnMajor)
        # Test the plotted instances
        self.assertEqual(2, len(self.axes.lines))
        # The no attacks live in lines[0]
        no_attacks = self.axes.lines[0].get_xydata()
        desired_no_attacks = self.X[self.y==0]
        np.testing.assert_array_equal(desired_no_attacks, no_attacks)
        # The attacks live in lines [1]
        attacks = self.axes.lines[1].get_xydata()
        desired_attacks = self.X[self.y==1]
        np.testing.assert_array_equal(desired_attacks, attacks)

    def test_plot_precision_recall_vs_threshold(self):
        sgd_clf = SGDClassifier(max_iter=5, random_state=13)
        y_scores = cross_val_predict(sgd_clf, self.X, self.y, cv=5, method='decision_function')
        precisions, recalls, thresholds = precision_recall_curve(self.y, y_scores)
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        # Test the plot legend
        self.assertEqual(2, len(self.axes.get_legend().get_texts()))
        precision_legend = self.axes.get_legend().get_texts()[0].get_text()
        recall_legend = self.axes.get_legend().get_texts()[1].get_text()
        self.assertEqual('Precision', precision_legend)
        self.assertEqual('Recall', recall_legend)
        # Test the plot label
        label_x = self.axes.get_xlabel()
        self.assertEqual('Threshold', label_x)
        # Test grid visibility
        # Although it is not the best practice to access a private variable,
        # it seems the only way to check the axis existance
        self.assertTrue(self.axes.xaxis._gridOnMinor and self.axes.xaxis._gridOnMajor)
        self.assertTrue(self.axes.yaxis._gridOnMinor and self.axes.yaxis._gridOnMajor)
        # Test the plotted instances
        self.assertEqual(2, len(self.axes.lines))
        # The precision line is in lines[0]
        precision_line = self.axes.lines[0].get_xydata()
        desired_precision_line = np.column_stack((thresholds, precisions[:-1]))
        np.testing.assert_array_equal(desired_precision_line, precision_line)
        # The recall line is in [1]
        recall_line = self.axes.lines[1].get_xydata()
        desired_recall_line = np.column_stack((thresholds, recalls[:-1]))
        np.testing.assert_array_equal(desired_recall_line, recall_line)

    def test_plot_predictions_for_logistic_regression(self):
        log_reg_clf = LogisticRegression(random_state=13, solver='liblinear')
        log_reg_clf.fit(self.X, self.y)
        x_axes = np.array([-2.5, 1.2])
        y_axes = np.array([-0.1, 2])
        plot_predictions_for_logistic_regression(log_reg_clf, np.concatenate((x_axes, y_axes)))
        # Test the plotted instances
        self.assertEqual(1, len(self.axes.lines))
        # The boundary line is in lines[0]
        boundary_line = self.axes.lines[0].get_xydata()
        desired_boundary_line = np.column_stack((x_axes,
            -(log_reg_clf.coef_[0][0] * x_axes + log_reg_clf.intercept_[0]) / log_reg_clf.coef_[0][1]))
        np.testing.assert_array_equal(desired_boundary_line, boundary_line)

    def test_plot_predictions_for_SVC(self):
        pass

    def test_plot_predictions_for_KNN(self):
        pass

    def tearDown(self):
        pass


class TestDataPreparation(unittest.TestCase):

    def test_split_train_and_test_sets(self):
        pass


class TestModelEvaluation(unittest.TestCase):

    def test_cross_validate_models(self):
        pass

    def test_get_cross_validate_scores(self):
        pass

    def test_evaluate_model_with_precision_and_recall(self):
        pass


if __name__ == '__main__':
    unittest.main()
