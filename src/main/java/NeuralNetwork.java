import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements SupervisedLearner {
  private List<Layer> layers;

  private double momentum;
  private double learningRate;

  public NeuralNetwork() {
    this.layers = new ArrayList<>();
  }

  public NeuralNetwork copy() {
    NeuralNetwork newNeuralNetwork = new NeuralNetwork();

    if (layers.size() > 0) {
      Layer firstLayer = layers.get(0);
      newNeuralNetwork.addFirstLayer(
          firstLayer.getLayerType(), firstLayer.getInputs(), firstLayer.getOutputs());

      for (int i = 1; i < layers.size(); i++) {
        Layer nextLayer = layers.get(i);
        newNeuralNetwork.addLayer(nextLayer.getLayerType(), nextLayer.getOutputs());
      }
    }

    newNeuralNetwork.setMomentum(momentum);
    newNeuralNetwork.setLearningRate(learningRate);

    return newNeuralNetwork;
  }

  public void setMomentum(double momentum) {
    this.momentum = momentum;
  }

  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  public List<Layer> getLayers() {
    return layers;
  }

  @Override
  public String name() {
    return getClass().getSimpleName();
  }

  private void checkLayers() {
    if (layers.size() == 0) {
      throw new IllegalStateException("This network has no layers.");
    }
  }

  public void addFirstLayer(Layer.LayerType layerType, int inputs, int outputs) {
    this.layers.clear();
    addLayer(layerType, inputs, outputs);
  }

  private void addLayer(Layer.LayerType layerType, int inputs, int outputs) {
    switch (layerType) {
      case LINEAR:
        layers.add(new LinearLayer(inputs, outputs));
        break;
      case TANH:
        layers.add(new TanhLayer(inputs));
        break;
      case LEAKY_RECTIFIER:
        layers.add(new LeakyRectifierLayer(inputs));
      case CONVOLUTION:
        // TODO implement
        break;
      default:
        throw new IllegalArgumentException("No implementation exists for this layer type.");
    }
  }

  public void addLayer(Layer.LayerType layerType, int outputs) {
    if (layers.isEmpty()) {
      throw new IllegalStateException("You must first add a layer with a specified number of inputs");
    }

    int previousOutputs = layers.get(layers.size() - 1).getOutputs();
    addLayer(layerType, previousOutputs, outputs);
  }

  public void initializeWeights() {
    layers.forEach(Layer::initialize);
  }

  public boolean isValid() {
    return layers.size() > 0;
  }

  public boolean isLinearNetwork() {
    return layers.size() == 1 && layers.get(0).getLayerType() == Layer.LayerType.LINEAR;
  }

  @Override
  public Vector predict(Vector in) {
    checkLayers();

    layers.get(0).activate(in);

    for (int i = 1; i < layers.size(); i++) {
      Vector previousActivation = layers.get(i - 1).getActivation();
      layers.get(i).activate(previousActivation);
    }

    return layers.get(layers.size() - 1).getActivation();
  }

  public void updateWeights() {
    for (Layer layer : layers) {
      layer.applyGradient(learningRate, momentum);
    }
  }

  public void backPropagate(Vector target) {
    checkLayers();

    Vector blame = target.copy();
    blame.addScaled(layers.get(layers.size() - 1).getActivation(), -1);
    layers.get(layers.size() - 1).setBlame(blame);

    for (int i = layers.size() - 1; i >= 1; i--) {
      blame = layers.get(i).backPropagate();
      layers.get(i - 1).setBlame(blame);
    }
  }

  public void updateGradient(Vector x) {
    checkLayers();

    Vector previousActivation = x;
    for (Layer layer : layers) {
      layer.updateGradient(previousActivation);
      previousActivation = layer.getActivation();
    }
  }

  public void printTopology() {
    for (int i = 0; i < layers.size(); i++) {
      System.out.printf("%d) %s\n", i, layers.get(i).topologyString());
    }
  }
}
