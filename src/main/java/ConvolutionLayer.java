import java.util.Arrays;

public class ConvolutionLayer extends ConnectedLayer<Tensor, Vector> {
  private int[] inputDimensions;
  private int[] filterDimensions;
  private int[] outputDimensions;

  public ConvolutionLayer(int[] inputDimensions, int[] filterDimensions, int[] outputDimensions) {
    super(Tensor.countElements(inputDimensions), Tensor.countElements(outputDimensions));

    this.inputDimensions = inputDimensions;
    this.filterDimensions = filterDimensions;
    this.outputDimensions = outputDimensions;

    normalizeDimensions();
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
  public void initialize() {
    super.initialize();

    setActivation(new Tensor(outputDimensions));
    setWeights(new Tensor(filterDimensions));
    setBias(new Vector(getWeights().getLastDimension()));

    final int filterElements = Tensor.countElements(filterDimensions);
    fillAll(() -> Math.max(0.01, 1.0 / filterElements) * Main.RANDOM.nextGaussian());
  }

  @Override
  public void setWeights(Tensor filter) {
    Tensor filterTensor = Tensor.asTensor(filter, filterDimensions);

    if (getWeights() != null && !Tensor.sizesEqual(filterTensor, getWeights())) {
      throw new IllegalArgumentException(
          "The new filter must match the existing filter dimensions.");
    }

    super.setWeights(filterTensor);
  }

  public void setBias(Vector bias) {
    if (bias.size() != filterDimensions[filterDimensions.length - 1]) {
      throw new IllegalArgumentException(
          "The bias length must be the same as the number of filters.");
    }

    super.setBias(bias);
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
      Tensor.convolvePerFilter(input, getWeights(), getActivation());
    } else {
      Tensor.convolve(input, getWeights(), getActivation());
    }

    addBiasesByLastDimension();

    return getActivation();
  }

  private void addBiasesByLastDimension() {
    Tensor[] activationLayers = getActivation().splitByLastDimension();

    for (int i = 0; i < activationLayers.length; i++) {
      activationLayers[i].addScalar(getBias().get(i));
    }
  }

  @Override
  Vector backPropagate() {
    Tensor result = new Tensor(inputDimensions);
    Tensor.convolvePerLayer(getWeights(), getBlame(), result, true);
    return result;
  }

  @Override
  void resetGradient() {
    if (getWeightsGradient() == null) {
      setWeightsGradient(new Tensor(filterDimensions));
    }

    if (getBiasGradient() == null) {
      setBiasGradient(new Vector(filterDimensions[filterDimensions.length - 1]));
    }

    super.resetGradient();
  }

  @Override
  void updateGradient(Vector x) {
    Tensor input = Tensor.asTensor(x, inputDimensions);

    Tensor.convolvePerFilter(input, getBlame(), getWeightsGradient());

    Tensor[] blameSlices = getBlame().splitByLastDimension();

    for (int i = 0; i < getBiasGradient().size(); i++) {
      Tensor blameSlice = blameSlices[i];
      getBiasGradient().set(i, blameSlice.reduce());
    }
  }
}
