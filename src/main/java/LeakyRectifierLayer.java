public class LeakyRectifierLayer extends Layer {

  public LeakyRectifierLayer(int inputs) {
    super(inputs, inputs);
  }

  public LeakyRectifierLayer copy() {
    return new LeakyRectifierLayer(getInputs());
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
    return getBlame().map((value) -> {
      if (value > 0) {
        return value;
      } else if (value == 0) {
        return 0.0;
      } else {
        return 0.01 * value;
      }
    });
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
