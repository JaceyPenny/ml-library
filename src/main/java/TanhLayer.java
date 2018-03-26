public class TanhLayer extends Layer {

  public TanhLayer(int inputs) {
    super(inputs, inputs);
  }

  @Override
  public LayerType getLayerType() {
    return LayerType.TANH;
  }

  @Override
  Vector activate(Vector x) {
    setActivation(x.map(Math::tanh));
    return getActivation();
  }

  @Override
  Vector backPropagate() {
    Vector previousBlame = new Vector(getBlame().size());
    for (int i = 0; i < previousBlame.size(); i++) {
      double newValue = getBlame().get(i) * (1.0 - getActivation().get(i) * getActivation().get(i));
      previousBlame.set(i, newValue);
    }

    return previousBlame;
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