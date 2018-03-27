public class MaxPooling2DLayer extends Layer {
  private int[] inputDimensions;
  private int[] outputDimensions;

  public MaxPooling2DLayer(int[] inputDimensions) {
    super(
        Tensor.countElements(inputDimensions),
        Tensor.countElements(inputDimensions) / 4);

    if (inputDimensions.length < 3) {
      throw new IllegalArgumentException(
          "Your input dimensions must have a dimensionality of 3 or greater.");
    }

    outputDimensions = new int[inputDimensions.length];
    System.arraycopy(inputDimensions, 0, outputDimensions, 0, inputDimensions.length);

    outputDimensions[0] /= 2;
    outputDimensions[1] /= 2;
  }

  @Override
  public LayerType getLayerType() {
    return LayerType.MAX_POOLING_2D;
  }

  @Override
  public MaxPooling2DLayer copy() {
    return new MaxPooling2DLayer(inputDimensions);
  }

  @Override
  Vector activate(Vector x) {
    Tensor input = new Tensor(x, inputDimensions);
    Tensor activation = new Tensor(getActivation(), outputDimensions);

    for (int k = 0; k < outputDimensions[2]; k++) {
      for (int j = 0; j < outputDimensions[1]; j++) {
        for (int i = 0; i < outputDimensions[0]; i++) {
          double value1 = input.get(i * 2, j * 2, k);
          double value2 = input.get(i * 2 + 1, j * 2, k);
          double value3 = input.get(i * 2, j * 2 + 1, k);
          double value4 = input.get(i * 2 + 1, j * 2 + 1, k);

          double max = Math.max(Math.max(value1, value2), Math.max(value3, value4));
          activation.set(max, i, j, k);
        }
      }
    }

    setActivation(activation);
    return getActivation();
  }

  @Override
  Vector backPropagate() {
    // TODO Implement
    return null;
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
