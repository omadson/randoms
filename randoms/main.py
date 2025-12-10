class GRADINGS(BaseDetector):
    """Implementation of GRADINGS.

    Parameters
    ----------
    a: string, optional (default='aaa')
        faz asadsda. Asdasd.

    Attributes
    ----------

    """
    def __init__(self, a='aaa'):
        super(GRADINGS, self).__init__(contamination=contamination)
        self.a = a
    
    def fit(self, X, y=None):
        X = check_array(X)
        self.detector_ = ...
        self.detector_.fit(X=X, y=y)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        # Invert outlier scores. Outliers comes with higher outlier scores
        return invert_order(self.detector_.decision_function(X))
