import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork extends SupervisedLearner {
    private int layerSize;

    private List<LayerLinear> layers;
    private List<Vector> layerWeights;

    public NeuralNetwork() {
        this(1);
    }

    public NeuralNetwork(int layerSize) {
        this.layers = new ArrayList<>();
        this.layerWeights = new ArrayList<>();
        this.layerSize = layerSize;
    }

    @Override
    String name() {
        return getClass().getSimpleName();
    }

    private void initializeLayers(int inputs, int outputs) {
        layers.clear();
        layerWeights.clear();

        for (int i = 0; i < layerSize; i++) {
            layers.add(new LayerLinear(inputs, outputs));
            layerWeights.add(new Vector(outputs + inputs * outputs));
        }
    }

    @Override
    void train(Matrix features, Matrix labels) {
        initializeLayers(features.cols(), labels.cols());

        for (int i = 0; i < layers.size(); i++) {
            LayerLinear layer = layers.get(i);
            Vector weights = layerWeights.get(i);

            layer.ordinaryLeastSquares(features, labels, weights);
        }
    }

    @Override
    Vector predict(Vector in) {
        if (layers.size() == 0) {
            throw new IllegalStateException("The network has not been trained yet.");
        }

        // For now, just use the first layer. I don't know what the implementation of future
        // neural network prediction needs to be
        LayerLinear layer = layers.get(0);
        Vector weights = layerWeights.get(0);

        layer.activate(weights, in);
        return layer.activation;
    }

    @Override
    double computeSumSquaredError(Matrix testFeatures, Matrix expectedLabels) {
        double sumSquaredError = 0;

        for (int i = 0; i < testFeatures.rows(); i++) {
            Vector x_i = testFeatures.row(i);
            Vector expected_y = expectedLabels.row(i);
            Vector calculated_y = predict(x_i);

            // Calculate squared error
            calculated_y.scale(-1);
            calculated_y.add(expected_y);
            double squaredError = calculated_y.squaredMagnitude();

            sumSquaredError += squaredError;
        }

        return sumSquaredError;
    }

    private void shuffleData(Matrix features, Matrix labels) {
        Random random = new Random();
        for (int i = features.rows(); i >= 2; i--) {
            int r = random.nextInt(i);
            features.swapRows(i - 1, r);
            labels.swapRows(i - 1, r);
        }
    }

    @Override
    double crossValidation(int folds, int repetitions, Matrix features, Matrix labels) {
        int dataRows = features.rows();
        int[] foldSizes = Matrix.computeFoldSizes(dataRows, folds);

        double totalError = 0;

        for (int r = 0; r < repetitions; r++) {
            shuffleData(features, labels);

            for (int i = 0; i < folds; i++) {
                Matrix X = Matrix.matrixWithoutFold(foldSizes, i, features);
                Matrix Y = Matrix.matrixWithoutFold(foldSizes, i, labels);

                Matrix test_X = Matrix.matrixFold(foldSizes, i, features);
                Matrix expected_Y = Matrix.matrixFold(foldSizes, i, labels);

                train(X, Y);

                double sumSquaredError = computeSumSquaredError(test_X, expected_Y);
                totalError += sumSquaredError;
            }
        }

        return Math.sqrt(totalError / repetitions / features.rows());
    }
}
