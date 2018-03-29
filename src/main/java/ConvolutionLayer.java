import java.util.Arrays;
import java.util.function.Supplier;

public class ConvolutionLayer extends Layer {
  private int[] inputDimensions;
  private int[] filterDimensions;
  private int[] outputDimensions;

  private Tensor filter;
  private Vector bias;

  private Tensor filterGradient;
  private Vector biasGradient;

  public ConvolutionLayer(int[] inputDimensions, int[] filterDimensions, int[] outputDimensions) {
    super(Tensor.countElements(inputDimensions), Tensor.countElements(outputDimensions));

    this.inputDimensions = inputDimensions;
    this.filterDimensions = filterDimensions;
    this.outputDimensions = outputDimensions;

    normalizeDimensions();

    setActivation(new Tensor(outputDimensions));
    filter = new Tensor(filterDimensions);
    bias = new Vector(filter.getLastDimension());

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

  // TODO Remove; This method only for testing
  public void setFilter(Tensor filter) {
    if (!Tensor.sizesEqual(filter, this.filter)) {
      throw new IllegalArgumentException(
          "The new filter must match the existing filter dimensions.");
    }

    this.filter = filter;
  }

  // TODO Remove; This is only for testing
  public void setBias(Vector bias) {
    if (bias.size() != filter.getLastDimension()) {
      throw new IllegalArgumentException(
          "The bias length must be the same as the number of filters.");
    }

    this.bias = bias;
  }

  // TODO Remove; This is only for testing
  public Tensor getGradient() {
    return filterGradient;
  }

  // TODO Remove; This is only for testing
  public Vector getBiasGradient() {
    return biasGradient;
  }

  // TODO Remove; This is only for testing
  public Tensor getFilter() {
    return filter;
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
  public Tensor getActivation() {
    return Tensor.asTensor(super.getActivation(), outputDimensions);
  }

  @Override
  public Tensor getBlame() {
    return Tensor.asTensor(super.getBlame(), outputDimensions);
  }

  @Override
  Vector activate(Vector x) {
    Tensor input = Tensor.asTensor(x, inputDimensions);

    getActivation().fill(0);

    if (input.getLastDimension() == 1) {
      Tensor.convolvePerFilter(input, filter, getActivation());
    } else {
      Tensor.convolve(input, filter, getActivation());
    }

    addBiasesByLastDimension();

    return getActivation();
  }

  private void addBiasesByLastDimension() {
    Tensor[] activationLayers = getActivation().splitByLastDimension();

    for (int i = 0; i < activationLayers.length; i++) {
      activationLayers[i].addScalar(bias.get(i));
    }
  }

  @Override
  Vector backPropagate() {
    Tensor result = new Tensor(inputDimensions);
    result.fill(0);
    Tensor.convolve(filter, getBlame(), result, true);
    return result;
  }

  @Override
  public void initialize() {
    final int filterElements = Tensor.countElements(filterDimensions);
    fillAll(() -> Main.RANDOM.nextGaussian() / filterElements);
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
      biasGradient = new Vector(filter.getLastDimension());
    }

    filterGradient.fill(0);
    biasGradient.fill(0);
  }

  @Override
  void updateGradient(Vector x) {
    Tensor input = Tensor.asTensor(x, inputDimensions);

    Tensor.convolvePerFilter(input, getBlame(), filterGradient);

    Tensor[] blameGradients = getBlame().splitByLastDimension();

    for (int i = 0; i < biasGradient.size(); i++) {
      Tensor singleBlameGradient = blameGradients[i];
      biasGradient.set(i, singleBlameGradient.reduce());
    }
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
