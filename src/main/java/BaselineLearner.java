// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class BaselineLearner extends SupervisedLearner {
    double[] mode;

    String name() {
        return "Baseline";
    }

    void train(Matrix features, Matrix labels) {
        mode = new double[labels.cols()];
        for (int i = 0; i < labels.cols(); i++) {
            if (labels.valueCount(i) == 0)
                mode[i] = labels.columnMean(i);
            else
                mode[i] = labels.mostCommonValue(i);
        }
    }

    Vector predict(Vector in) {
        return new Vector(mode);
    }

    @Override
    double computeSumSquaredError(Matrix testFeatures, Matrix expectedLabels) {
        return 0;
    }

    @Override
    double crossValidation(int folds, int repetitions, Matrix features, Matrix labels) {
        return 0;
    }
}
