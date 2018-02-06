import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork extends SupervisedLearner {
  private int layerSize;

  private List<LayerLinear> layers;
  private List<Vector> allWeights;

  public NeuralNetwork() {
    this(1);
  }

  public NeuralNetwork(int layerSize) {
    this.layers = new ArrayList<>();
    this.layerSize = layerSize;
    this.allWeights = new ArrayList<>();
  }

  @Override
  String name() {
    return getClass().getSimpleName();
  }

  private void initializeLayers(int inputs, int outputs) {
    layers.clear();

    for (int i = 0; i < layerSize; i++) {
      allWeights.add(new Vector(outputs + inputs * outputs));
    }

    for (int i = 0; i < layerSize; i++) {
      layers.add(new LayerLinear(inputs, outputs));
    }
  }

  @Override
  void train(Matrix features, Matrix labels) {
    initializeLayers(features.cols(), labels.cols());

    for (int i = 0; i < layers.size(); i++) {
      layers.get(i).ordinaryLeastSquares(features, labels, allWeights.get(i));
    }
  }

  @Override
  Vector predict(Vector in) {
    if (layers.size() == 0) {
      throw new IllegalStateException("The network has not been trained yet.");
    }

    layers.get(0).activate(allWeights.get(0), in);

    for (int i = 1; i < layers.size(); i++) {
      Vector previousActivation = layers.get(i - 1).getActivation();
      layers.get(i).activate(allWeights.get(i), previousActivation);
    }

    return layers.get(layers.size() - 1).getActivation();
  }
}
