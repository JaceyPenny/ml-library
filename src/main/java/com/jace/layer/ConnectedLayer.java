package com.jace.layer;

import com.jace.math.Spatial;

import java.util.function.Supplier;

@SuppressWarnings({"unchecked", "WeakerAccess"})
public abstract class ConnectedLayer<T extends Spatial, U extends Spatial> extends Layer {
  public enum RegularizationType {
    NONE, L1, L2
  }

  private T weights;
  private U bias;

  private T weightsGradient;
  private U biasGradient;

  private RegularizationType regularizationType = RegularizationType.NONE;
  private double regularizationAmount = 0;

  ConnectedLayer(int inputs, int outputs) {
    super(inputs, outputs);
  }

  @Override
  public void initialize() {
    resetGradient();
  }

  RegularizationType getRegularizationType() {
    return regularizationType;
  }

  public void setRegularizationType(RegularizationType regularizationType) {
    this.regularizationType = regularizationType;
  }

  double getRegularizationAmount() {
    return regularizationAmount;
  }

  public void setRegularizationAmount(double regularizationAmount) {
    this.regularizationAmount = regularizationAmount;
  }

  @Override
  void resetGradient() {
    weightsGradient.fill(0);
    biasGradient.fill(0);
  }

  void fillAll(Supplier<Double> supplier) {
    getWeights().fill(supplier);
    getBias().fill(supplier);
  }

  @Override
  void applyGradient(double learningRate) {
    applyGradient(learningRate, 0);
  }

  @Override
  public void applyGradient(double learningRate, double momentum) {
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
