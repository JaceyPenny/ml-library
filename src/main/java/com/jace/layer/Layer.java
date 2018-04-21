package com.jace.layer;

import com.jace.math.Vector;

public abstract class Layer {
  public enum LayerType {
    LINEAR, TANH, LEAKY_RECTIFIER, CONVOLUTION, ASSIGNMENT_5, ASSIGNMENT_5_ACTIVATION, MAX_POOLING_2D
  }

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

  public abstract Layer copy();

  public abstract void initialize();

  public int getInputs() {
    return inputs;
  }

  public int getOutputs() {
    return outputs;
  }

  public abstract Vector activate(Vector x);

  protected void setActivation(Vector vector) {
    this.activation = vector;
  }

  public Vector getActivation() {
    return activation;
  }

  public void setBlame(Vector blame) {
    this.blame = blame;
  }

  public Vector getBlame() {
    return blame;
  }

  public abstract Vector backPropagate();

  abstract void resetGradient();

  public abstract void updateGradient(Vector x);

  abstract void applyGradient(double learningRate);

  public abstract void applyGradient(double learningRate, double momentum);

  public String topologyString() {
    String name = getLayerType().toString();
    int weights = (this instanceof LinearLayer) ? this.getInputs() * getOutputs() + getOutputs() : 0;

    return String.format("[%s: %d->%d, Weights=%d]", name, getInputs(), getOutputs(), weights);
  }
}
