import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork extends SupervisedLearner {
  private int inputs;

  private List<Layer> layers;

  public NeuralNetwork(int inputs) {
    this.inputs = inputs;
    this.layers = new ArrayList<>();
  }

  @Override
  String name() {
    return getClass().getSimpleName();
  }

  private void checkLayers() {
    if (layers.size() == 0) {
      throw new IllegalStateException("This network has no layers.");
    }
  }

  private void addLayer(LayerType layerType, int outputs) {
    int previousOutputs =
        (layers.size() == 0) ?
            inputs :
            layers.get(layers.size() - 1).getOutputs();


    switch (layerType) {
      case LINEAR:
        layers.add(new LayerLinear(previousOutputs, outputs));
        break;
      case TANH:
        // TODO Implement Tanh Layer
        break;
      default:
        throw new NotImplementedException();
    }
  }

  @Override
  void train(Matrix features, Matrix labels) {
    checkLayers();

    // TODO Update train methodology
  }

  @Override
  Vector predict(Vector in) {
    checkLayers();

    layers.get(0).activate(in);

    for (int i = 1; i < layers.size(); i++) {
      Vector previousActivation = layers.get(i - 1).getActivation();
      layers.get(i).activate(previousActivation);
    }

    return layers.get(layers.size() - 1).getActivation();
  }

  void backPropagate(Vector weights, Vector target) {
    checkLayers();

    Vector blame = Vector.copy(target);
    blame.addScaled(layers.get(layers.size() - 1).getActivation(), -1);
    layers.get(layers.size() - 1).setBlame(blame);

    for (int i = layers.size() - 2; i > 0; i++) {
      Layer layer = layers.get(i);

      layer.backPropagate(blame);
      blame = layer.getBlame();
    }
  }

  void updateGradient() {
    checkLayers();

    // TODO Update the gradient
  }
}
