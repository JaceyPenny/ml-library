public class LeakyRectifierLayer extends ActivationLayer {

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
    Vector result = new Vector(getInputs());

    for (int i = 0; i < getInputs(); i++) {
      double newValue = getBlame().get(i);

      if (getActivation().get(i) == 0) {
        newValue = 0;
      } else if (getActivation().get(i) < 0) {
        newValue *= 0.01;
      }

      result.set(i, newValue);
    }

    return result;
  }
}
