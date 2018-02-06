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

  public int getInputs() {
    return inputs;
  }

  public int getOutputs() {
    return outputs;
  }

  abstract void activate(Vector weights, Vector x);

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

  abstract void backPropagate(Vector weights, Vector previousBlame);

  abstract void updateGradient(Vector x, Vector gradient);
}
