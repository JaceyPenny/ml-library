import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork extends SupervisedLearner {
    private int layerSize;

    private List<LayerLinear> layers;
    private Vector weights;

    public NeuralNetwork() {
        this(1);
    }

    public NeuralNetwork(int layerSize) {
        this.layers = new ArrayList<>();
        this.layerSize = layerSize;
    }

    @Override
    String name() {
        return getClass().getSimpleName();
    }

    private void initializeLayers(int inputs, int outputs) {
        layers.clear();
        weights = new Vector(outputs + inputs * outputs);

        for (int i = 0; i < layerSize; i++) {
            layers.add(new LayerLinear(inputs, outputs));
        }
    }

    @Override
    void train(Matrix features, Matrix labels) {
        initializeLayers(features.cols(), labels.cols());

        for (LayerLinear layer : layers) {
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

        layer.activate(weights, in);
        return layer.getActivation();
    }
}
