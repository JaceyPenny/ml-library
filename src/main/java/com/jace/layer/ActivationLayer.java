package com.jace.layer;

import com.jace.math.Vector;

public abstract class ActivationLayer extends Layer {
  ActivationLayer(int inputs, int outputs) {
    super(inputs, outputs);
  }

  @Override
  public void initialize() {

  }

  @Override
  void resetGradient() {

  }

  @Override
  public void updateGradient(Vector x) {

  }

  @Override
  void applyGradient(double learningRate) {

  }

  @Override
  public void applyGradient(double learningRate, double momentum) {

  }
}
