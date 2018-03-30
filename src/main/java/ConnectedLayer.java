import java.util.function.Supplier;

public abstract class ConnectedLayer<T extends Spatial, U extends Spatial> extends Layer {
  private T weights;
  private U bias;

  private T weightsGradient;
  private U biasGradient;

  ConnectedLayer(int inputs, int outputs) {
    super(inputs, outputs);
  }

  @Override
  public void initialize() {
    resetGradient();
  }

  @Override
  void resetGradient() {
    weightsGradient.fill(0);
    biasGradient.fill(0);
  }

  protected void fillAll(Supplier<Double> supplier) {
    getWeights().fill(supplier);
    getBias().fill(supplier);
  }

  @Override
  void applyGradient(double learningRate) {
    applyGradient(learningRate, 0);
  }

  @Override
  void applyGradient(double learningRate, double momentum) {
    getWeights().addScaled(getWeightsGradient(), learningRate);
    getBias().addScaled(getBiasGradient(), learningRate);

    getWeightsGradient().scale(momentum);
    getBiasGradient().scale(momentum);
  }

  public T getWeights() {
    return weights;
  }

  public void setWeights(T weights) {
    this.weights = weights;
  }

  public U getBias() {
    return bias;
  }

  public void setBias(U bias) {
    this.bias = bias;
  }

  public T getWeightsGradient() {
    return weightsGradient;
  }

  public void setWeightsGradient(T weightsGradient) {
    this.weightsGradient = weightsGradient;
  }

  public U getBiasGradient() {
    return biasGradient;
  }

  public void setBiasGradient(U biasGradient) {
    this.biasGradient = biasGradient;
  }
}
