public abstract class ActivationLayer extends Layer {
  ActivationLayer(int inputs, int outputs) {
    super(inputs, outputs);
  }

  @Override
  public void initialize() {

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
