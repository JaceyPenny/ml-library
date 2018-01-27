import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork extends SupervisedLearner {
    private int layerSize;

    private List<LayerLinear> layers;
    private List<Vector> layerWeights;

    public NeuralNetwork(int layerSize) {
        this.layers = new ArrayList<>();
        this.layerWeights = new ArrayList<>();
        this.layerSize = layerSize;
    }

    @Override
    String name() {
        return "NeuralNetwork";
    }

    private void initializeLayers(int inputs, int outputs) {
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
    double computeSumSquaredError() {
        return 0;
    }
}
