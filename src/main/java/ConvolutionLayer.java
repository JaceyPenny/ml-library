public class ConvolutionLayer extends Layer {

  public ConvolutionLayer(int[] inputDimensions, int[] filterDimensions, int[] outputDimensions) {
    super(Tensor.countElements(inputDimensions), Tensor.countElements(outputDimensions));

  }

  @Override
  public LayerType getLayerType() {
    return LayerType.CONVOLUTION;
  }

  @Override
  Vector activate(Vector x) {
    return null;
  }

  @Override
  Vector backPropagate() {
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
