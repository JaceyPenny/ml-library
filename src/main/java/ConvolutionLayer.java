import java.util.Arrays;
import java.util.function.Supplier;

public class ConvolutionLayer extends Layer {

  private int[] inputDimensions;
  private int[] filterDimensions;
  private int[] outputDimensions;

  private Tensor filter;
  private Tensor bias;

  private Tensor filterGradient;
  private Tensor biasGradient;

  public ConvolutionLayer(int[] inputDimensions, int[] filterDimensions, int[] outputDimensions) {
    super(Tensor.countElements(inputDimensions), Tensor.countElements(outputDimensions));

    this.inputDimensions = inputDimensions;
    this.filterDimensions = filterDimensions;
    this.outputDimensions = outputDimensions;

    normalizeDimensions();

    setActivation(new Tensor(outputDimensions));
    filter = new Tensor(filterDimensions);
    bias = new Tensor(outputDimensions);

    initialize();
    resetGradient();
  }

  private void normalizeDimensions() {
    int maxDimensions = Math.max(
        Math.max(inputDimensions.length, filterDimensions.length),
        outputDimensions.length);

    int[] inputDimensions = new int[maxDimensions];
    int[] filterDimensions = new int[maxDimensions];
    int[] outputDimensions = new int[maxDimensions];

    // Set the size of each dimension to 1
    Arrays.fill(inputDimensions, 1);
    Arrays.fill(filterDimensions, 1);
    Arrays.fill(outputDimensions, 1);

    System.arraycopy(this.inputDimensions, 0, inputDimensions, 0, this.inputDimensions.length);
    System.arraycopy(this.filterDimensions, 0, filterDimensions, 0, this.filterDimensions.length);
    System.arraycopy(this.outputDimensions, 0, outputDimensions, 0, this.outputDimensions.length);

    this.inputDimensions = inputDimensions;
    this.filterDimensions = filterDimensions;
    this.outputDimensions = outputDimensions;
  }

  @Override
  public ConvolutionLayer copy() {
    return new ConvolutionLayer(inputDimensions, filterDimensions, outputDimensions);
  }

  @Override
  public LayerType getLayerType() {
    return LayerType.CONVOLUTION;
  }

  @Override
  Vector activate(Vector x) {
    Tensor input = new Tensor(x, inputDimensions);

    Tensor.convolve(input, filter, (Tensor) getActivation());
    getActivation().add(bias);
    return getActivation();
  }

  @Override
  Vector backPropagate() {
    Tensor result = new Tensor(inputDimensions);
    Tensor blameTensor = new Tensor(getBlame(), outputDimensions);

    Tensor.convolve(blameTensor, filter, result, true);
    return result;
  }

  @Override
  public void initialize() {
    fillAll(() ->
        Math.max(0.03, 1.0 / Tensor.countElements(inputDimensions)) * Main.RANDOM.nextGaussian());
  }

  private void fillAll(Supplier<Double> supplier) {
    filter.fill(supplier);
    bias.fill(supplier);
  }

  @Override
  void resetGradient() {
    if (filterGradient == null) {
      filterGradient = new Tensor(filterDimensions);
    }

    if (biasGradient == null) {
      biasGradient = new Tensor(outputDimensions);
    }

    filterGradient.fill(0);
    biasGradient.fill(0);
  }

  @Override
  void updateGradient(Vector x) {
    Tensor input = new Tensor(x, inputDimensions);
    Tensor blameTensor = new Tensor(getBlame(), outputDimensions);

    Tensor.convolve(input, blameTensor, filterGradient);
    biasGradient.add(getBlame());
  }

  @Override
  void applyGradient(double learningRate) {
    applyGradient(learningRate, 0);
  }

  @Override
  void applyGradient(double learningRate, double momentum) {
    filter.addScaled(filterGradient, learningRate);
    bias.addScaled(biasGradient, learningRate);

    filterGradient.scale(momentum);
    biasGradient.scale(momentum);
  }
}
