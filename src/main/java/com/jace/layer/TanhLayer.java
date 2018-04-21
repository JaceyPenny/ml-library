package com.jace.layer;

import com.jace.math.Vector;

public class TanhLayer extends ActivationLayer {

  public TanhLayer(int inputs) {
    super(inputs, inputs);
  }

  @Override
  public TanhLayer copy() {
    return new TanhLayer(getInputs());
  }

  @Override
  public Layer.LayerType getLayerType() {
    return Layer.LayerType.TANH;
  }

  @Override
  public Vector activate(Vector x) {
    setActivation(x.map(Math::tanh));
    return getActivation();
  }

  @Override
  public Vector backPropagate() {
    Vector previousBlame = new Vector(getBlame().size());
    for (int i = 0; i < previousBlame.size(); i++) {
      double newValue = getBlame().get(i) * (1.0 - getActivation().get(i) * getActivation().get(i));
      previousBlame.set(i, newValue);
    }

    return previousBlame;
  }
}
