public class LeakyRectifierLayer extends Layer {

  public LeakyRectifierLayer(int inputs) {
    super(inputs, inputs);
  }

  @Override
  public LayerType getLayerType() {
    return LayerType.LEAKY_RECTIFIER;
  }

  @Override
  Vector activate(Vector x) {
    setActivation(x.map((value) -> (value >= 0) ? value : 0.01 * value));
    return getActivation();
  }

  @Override
  Vector backPropagate() {
    return getBlame().map((value) -> (value >= 0) ? 1 : 0.01);
  }

  @Override
  void resetGradient() {

  }

  @Override
  void updateGradient(Vector x) {

  }

  @Override
  void applyGradient(double learningRate) {

  }

  @Override
  void applyGradient(double learningRate, double momentum) {

  }
}
