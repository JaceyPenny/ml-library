abstract class Layer {
  private int inputs;
  private int outputs;

  private Vector activation;
  private Vector blame;

  Layer(int inputs, int outputs) {
    this.inputs = inputs;
    this.outputs = outputs;

    activation = new Vector(outputs);
    blame = new Vector(outputs);
  }

  public abstract LayerType getLayerType();

  public int getInputs() {
    return inputs;
  }

  public int getOutputs() {
    return outputs;
  }

  abstract Vector activate(Vector x);

  protected void setActivation(Vector vector) {
    this.activation = vector;
  }

  public Vector getActivation() {
    return activation;
  }

  protected void setBlame(Vector blame) {
    this.blame = blame;
  }

  public Vector getBlame() {
    return blame;
  }

  abstract Vector backPropagate();

  abstract void resetGradient();

  abstract void updateGradient(Vector x);

  abstract void applyGradient(double learningRate);

  abstract void applyGradient(double learningRate, double momentum);

  public String topologyString() {
    String name = getLayerType().toString();
    int weights = (this instanceof LayerLinear) ? this.getInputs() * getOutputs() + getOutputs() : 0;

    return String.format("[%s: %d->%d, Weights=%d]", name, getInputs(), getOutputs(), weights);
  }
}
