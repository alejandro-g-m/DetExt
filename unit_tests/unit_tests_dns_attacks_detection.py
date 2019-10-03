import unittest
import sys
sys.path.append('..') # The code is in the parent directory

from sklearn.datasets import make_classification
from dns_attacks_detection import *


class TestPlotting(unittest.TestCase):

    @staticmethod
    def is_point_within_polygon(point_x, point_y, vert_x, vert_y):
        """
        Helper function that calculates if a point is inside a poligon.
        Used to test if the contours are drawn properly.
        Taken from: https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
        """
        n_vert = len(vert_x)
        result = False
        i = 0
        j = n_vert-1
        while i < n_vert:
            if (((vert_y[i]>point_y) != (vert_y[j]>point_y)) and
            (point_x < (vert_x[j]-vert_x[i]) * (point_y-vert_y[i]) / (vert_y[j]-vert_y[i]) + vert_x[i])):
                result = not result
            j = i
            i += 1
        return result

    def setUp(self):
        self.X, self.y = make_classification(n_samples=10, n_features=2,
        n_redundant=0, n_informative=2, random_state=13, n_clusters_per_class=1)
        self.figure, self.axes = plt.subplots()

    def test_plot_dataset(self):
        plot_dataset(self.X, self.y, "first_feature", "second_feature")
        # Test the plot legend
        # (are there 2 legends?)
        self.assertEqual(2, len(self.axes.get_legend().get_texts()))
        no_attack_legend = self.axes.get_legend().get_texts()[0].get_text()
        attack_legend = self.axes.get_legend().get_texts()[1].get_text()
        # (do the legends have the proper text?)
        self.assertEqual('No attack', no_attack_legend)
        self.assertEqual('Attack', attack_legend)
        # Test the plot labels
        label_x = self.axes.get_xlabel()
        label_y = self.axes.get_ylabel()
        # (do the labels have the proper text?)
        self.assertEqual('first_feature', label_x)
        self.assertEqual('second_feature', label_y)
        # Test grid visibility
        # Although it is not the best practice to access a private variable,
        # it seems the only way to check the axis existance
        # (are the x and y axis shown on the minor and major lines?)
        self.assertTrue(self.axes.xaxis._gridOnMinor and self.axes.xaxis._gridOnMajor)
        self.assertTrue(self.axes.yaxis._gridOnMinor and self.axes.yaxis._gridOnMajor)
        # Test the plotted instances
        # (are there 2 plots, one for attacks and one for no attacks?)
        self.assertEqual(2, len(self.axes.lines))
        # The no attacks live in lines[0]
        no_attacks = self.axes.lines[0].get_xydata()
        desired_no_attacks = self.X[self.y==0]
        # (do the no attacks match the no attacks from the dataset?)
        np.testing.assert_array_equal(desired_no_attacks, no_attacks)
        # The attacks live in lines [1]
        attacks = self.axes.lines[1].get_xydata()
        desired_attacks = self.X[self.y==1]
        # (do the attacks match the attacks from the dataset?)
        np.testing.assert_array_equal(desired_attacks, attacks)

    def test_plot_precision_recall_vs_threshold(self):
        sgd_clf = SGDClassifier(max_iter=5, random_state=13)
        y_scores = cross_val_predict(sgd_clf, self.X, self.y, cv=5, method='decision_function')
        precisions, recalls, thresholds = precision_recall_curve(self.y, y_scores)
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        # Test the plot legend
        # (are there 2 legends?)
        self.assertEqual(2, len(self.axes.get_legend().get_texts()))
        precision_legend = self.axes.get_legend().get_texts()[0].get_text()
        recall_legend = self.axes.get_legend().get_texts()[1].get_text()
        # (do the legends have the proper text?)
        self.assertEqual('Precision', precision_legend)
        self.assertEqual('Recall', recall_legend)
        # Test the plot label
        # (does the label have the proper text?)
        label_x = self.axes.get_xlabel()
        self.assertEqual('Threshold', label_x)
        # Test grid visibility
        # Although it is not the best practice to access a private variable,
        # it seems the only way to check the axis existance
        # (are the x and y axis shown on the minor and major lines?)
        self.assertTrue(self.axes.xaxis._gridOnMinor and self.axes.xaxis._gridOnMajor)
        self.assertTrue(self.axes.yaxis._gridOnMinor and self.axes.yaxis._gridOnMajor)
        # Test the plotted instances
        # (are there 2 plots, one for the precision line and one for the recall line?)
        self.assertEqual(2, len(self.axes.lines))
        # The precision line is in lines[0]
        precision_line = self.axes.lines[0].get_xydata()
        desired_precision_line = np.column_stack((thresholds, precisions[:-1]))
        # (is the precision line drawn according to the precisions of the clasifier?)
        np.testing.assert_array_equal(desired_precision_line, precision_line)
        # The recall line is in [1]
        recall_line = self.axes.lines[1].get_xydata()
        desired_recall_line = np.column_stack((thresholds, recalls[:-1]))
        # (is the recall line drawn according to the recalls of the clasifier?)
        np.testing.assert_array_equal(desired_recall_line, recall_line)

    def test_plot_predictions_for_logistic_regression(self):
        log_reg_clf = LogisticRegression(random_state=13, solver='liblinear')
        log_reg_clf.fit(self.X, self.y)
        x_axes = np.array([-2.5, 1.2])
        y_axes = np.array([-0.1, 2])
        plot_predictions_for_logistic_regression(log_reg_clf, np.concatenate((x_axes, y_axes)))
        # Test the plotted boundary line
        # (is there just one line plotted?)
        self.assertEqual(1, len(self.axes.lines))
        # The boundary line is in lines[0]
        boundary_line = self.axes.lines[0].get_xydata()
        desired_boundary_line = np.column_stack((x_axes,
            -(log_reg_clf.coef_[0][0] * x_axes + log_reg_clf.intercept_[0]) / log_reg_clf.coef_[0][1]))
        # (does the line match the values from the clasifier?)
        np.testing.assert_array_equal(desired_boundary_line, boundary_line)

    def test_plot_predictions_for_SVC(self):
        svm_clf_poly = SVC(kernel='poly', random_state=13, gamma='auto',
        degree=3, coef0=1, C=5)
        svm_clf_poly.fit(self.X, self.y)
        x_axes = np.array([-2.5, 1.2])
        y_axes = np.array([-0.1, 2])
        plot_predictions_for_SVC(svm_clf_poly, np.concatenate((x_axes, y_axes)))
        y_pred = svm_clf_poly.predict(self.X)
        # Test if the predictions are displayed correctly
        contour_0 = self.axes.collections[0].get_paths()
        vertices_x = contour_0[0].vertices[:,0]
        vertices_y = contour_0[0].vertices[:,1]
        for X, prediction in zip(self.X, y_pred):
            if prediction == 1:
                # If it is predicted as an attack it should be out of the contour
                self.assertFalse(
                self.is_point_within_polygon(X[0], X[1], vertices_x, vertices_y))
            else:
                # If it is predicted as a no attack it should be in the contour
                self.assertTrue(
                self.is_point_within_polygon(X[0], X[1], vertices_x, vertices_y))

    def test_plot_predictions_for_KNN(self):
        """
        There does not seem to be a straight forward way to test
        'plot_predictions_for_KNN' because this function is only plotting
        a map of colours. Therefore, this test is skipped.
        """
        pass



class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(
        n_samples=1000, n_features=5, n_redundant=0, n_informative=2,
        random_state=13, n_clusters_per_class=1)
        self.df = pd.DataFrame(np.concatenate((self.X, np.reshape(self.y, (-1,1))), axis=1))
        self.target = 5

    def test_split_train_and_test_sets(self):
        """
        The purpose of the test is to find that the train and test sets have
        similar proportions regarding the target variable and that there
        sizes of both sets are correct.
        """
        # Test dataframe
        train_set, test_set = split_train_and_test_sets(self.df, self.target)
        train_prop = train_set[self.target].value_counts() / len(train_set)
        test_prop = test_set[self.target].value_counts() / len(test_set)
        # Test similar proportions of each target variable
        np.testing.assert_almost_equal(train_prop[0], test_prop[0], decimal=2)
        np.testing.assert_almost_equal(train_prop[1], test_prop[1], decimal=2)
        # Test that sizes of train and test sets are the correct ones
        np.testing.assert_almost_equal(
        len(train_set) / (len(train_set) + len(test_set)), 0.8, decimal=2)
        np.testing.assert_almost_equal(
        len(test_set) / (len(train_set) + len(test_set)), 0.2, decimal=2)

        # Test dataframe with changed sizes
        train_set, test_set = split_train_and_test_sets(self.df, self.target, 0.5)
        train_prop = train_set[self.target].value_counts() / len(train_set)
        test_prop = test_set[self.target].value_counts() / len(test_set)
        # Test similar proportions of each target variable
        np.testing.assert_almost_equal(train_prop[0], test_prop[0], decimal=2)
        np.testing.assert_almost_equal(train_prop[1], test_prop[1], decimal=2)
        # Test that sizes of train and test sets are the correct ones
        np.testing.assert_almost_equal(
        len(train_set) / (len(train_set) + len(test_set)), 0.5, decimal=2)
        np.testing.assert_almost_equal(
        len(test_set) / (len(train_set) + len(test_set)), 0.5, decimal=2)



class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(
        n_samples=1000, n_features=5, n_redundant=0, n_informative=2,
        random_state=13, n_clusters_per_class=1)
        sgd_clf = SGDClassifier(max_iter=5, random_state=13)
        log_reg_clf = LogisticRegression(random_state=13, solver='liblinear')
        svm_clf_poly = SVC(kernel='poly', random_state=13, gamma='auto',
        degree=3, coef0=1, C=5)
        self.models = [sgd_clf, log_reg_clf, svm_clf_poly]
        self.names = ['SGDClassifier', 'LogisticRegression', 'SVC Poly']
        self.scoring = ['precision', 'recall', 'f1']
        # Test cross validation with no parameters
        self.cv_1 = cross_validate_models(self.models, self.X, self.y, self.scoring)
        # Test cross validation with parameters
        self.cv_2 = cross_validate_models(self.models, self.X, self.y, self.scoring,
        cv=7, return_train_score=True)

    def test_cross_validate_models(self):
        """
        Most of the functionality relies in calling the cross_validate function
        so the test will focus in checking that there are results for every
        model and every model has the required scores.
        """
        # Use cv_1 with no parameters
        # Test results for the three models
        self.assertEqual(len(self.models), len(self.cv_1))
        for model in self.cv_1:
            for score in self.scoring:
                score_string = 'test_' + score
                # Test scores are present
                self.assertTrue(score_string in model)
                # Test every score results for every fold (5)
                self.assertEqual(5, len(model[score_string]))
        # Use cv_2 with parameters
        # Test results for the three models
        self.assertEqual(len(self.models), len(self.cv_2))
        for model in self.cv_2:
            for score in self.scoring:
                test_score_string = 'test_' + score
                train_score_string = 'train_' + score
                # Test scores are present (test and train)
                self.assertTrue(test_score_string in model)
                self.assertTrue(train_score_string in model)
                # Test every score results for every fold (7)
                self.assertEqual(7, len(model[test_score_string]))

    def test_get_cross_validate_scores(self):
        cv_sc = get_cross_validate_scores(self.cv_1, self.names, self.scoring)
        # Test all models are present
        self.assertEqual(len(self.models), len(cv_sc))
        for model, scores in cv_sc.items():
            self.assertTrue(model in self.names)
            # Test all scores are present
            self.assertEqual(len(self.scoring), len(scores))
            for score, _ in scores.items():
                self.assertTrue(score in self.scoring)

    def test_evaluate_model_with_precision_and_recall(self):
        # Create simple classification problem
        y = np.concatenate((np.zeros(50), np.ones(50)))
        X = np.c_[y,y]
        model = self.models[0].fit(X, y)
        # Evaluate model
        scores = evaluate_model_with_precision_and_recall(model, X, y)
        self.assertEqual(3, len(scores))
        for score in scores: np.testing.assert_almost_equal(1.0, score, decimal=2)
        # Evaluate model with inverse data so all the predictions are wrong
        scores = evaluate_model_with_precision_and_recall(model, X, np.flip(y, 0))
        self.assertEqual(3, len(scores))
        for score in scores: np.testing.assert_almost_equal(0, score, decimal=2)
        # Evaluate model with half "wrong" data
        scores = evaluate_model_with_precision_and_recall(model, X, np.ones(100))
        self.assertEqual(3, len(scores))
        np.testing.assert_almost_equal(1, scores[0], decimal=2)
        np.testing.assert_almost_equal(0.5, scores[1], decimal=2)
        np.testing.assert_almost_equal(0.66, scores[2], decimal=2)

    def test_print_scores(self):
        # Generate a string and check that the returned string is the same
        models_and_scores = (
        {'model1': {'score1': 1.0, 'score2': 0.5, 'score3': 0},
        'model2': {'score1': 0.7, 'score2': 0.3, 'score3': 0.6}})
        scores = print_scores('my_vector', models_and_scores)
        desired = color.BOLD + 'my_vector' + color.END + "\n"
        desired += color.UNDERLINE + 'model1' + color.END + "\n"
        desired += tabulate([(sc_name, sc_result)
        for sc_name, sc_result in models_and_scores['model1'].items()], tablefmt="plain") + "\n"
        desired += color.UNDERLINE + 'model2' + color.END + "\n"
        desired += tabulate([(sc_name, sc_result)
        for sc_name, sc_result in models_and_scores['model2'].items()], tablefmt="plain") + "\n"
        self.assertEqual(desired, scores)



if __name__ == '__main__':
    unittest.main()
