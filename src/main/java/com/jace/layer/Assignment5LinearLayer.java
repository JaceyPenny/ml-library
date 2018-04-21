package com.jace.layer;

import com.jace.math.Matrix;
import com.jace.math.Vector;

public class Assignment5LinearLayer extends LinearLayer {

  public Assignment5LinearLayer() {
    super(1, 101);
  }

  @Override
  public void initialize() {
    super.initialize();

    setWeights(new Matrix(101, 1));
    setBias(new Vector(101));

    double pi = Math.PI;
    for (int i = 0; i < 50; i++) {
      getWeights().set(i, 0, (i + 1) * 2 * pi);
      getBias().set(i, pi);
    }

    for (int i = 50; i < 100; i++) {
      getWeights().set(i, 0, (i - 49) * 2 * pi);
      getBias().set(i, pi / 2);
    }

    getWeights().set(100, 0.01);
    getBias().set(100, 0);
  }

  @Override
  public Layer.LayerType getLayerType() {
    return Layer.LayerType.ASSIGNMENT_5;
  }

  @Override
  public LinearLayer copy() {
    return new Assignment5LinearLayer();
  }
}
