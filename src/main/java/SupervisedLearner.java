// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

abstract class SupervisedLearner {
    /**
     * Return the name of this learner
     */
    abstract String name();

    /**
     * Train this supervised learner
     */
    abstract void train(Matrix features, Matrix labels);

    /**
     * Make a prediction
     */
    abstract Vector predict(Vector in);

    /**
     * Computes the sum squared error for this learner.
     */
    abstract double computeSumSquaredError(Matrix testFeatures, Matrix expectedLabels);

    abstract double crossValidation(int folds, int repetitions, Matrix features, Matrix labels);
    /**
     * Measures the misclassificaions with the provided test data
     */
    int countMisclassifications(Matrix features, Matrix labels) {
        if (features.rows() != labels.rows())
            throw new IllegalArgumentException("Mismatching number of rows");
        int mis = 0;
        for (int i = 0; i < features.rows(); i++) {
            Vector feature = features.row(i);
            Vector prediction = predict(feature);
            Vector lab = labels.row(i);
            for (int j = 0; j < lab.size(); j++) {
                if (prediction.get(j) != lab.get(j))
                    mis++;
            }
        }
        return mis;
    }
}
