package com.jace.layer;

import com.jace.math.Vector;

public class Assignment5ActivationLayer extends ActivationLayer {

  public Assignment5ActivationLayer() {
    super(101, 101);
  }

  @Override
  public Layer.LayerType getLayerType() {
    return Layer.LayerType.ASSIGNMENT_5_ACTIVATION;
  }

  @Override
  public Layer copy() {
    return new Assignment5ActivationLayer();
  }

  @Override
  public Vector activate(Vector x) {
    Vector map = x.map(Math::sin);
    map.set(100, x.get(100));

    setActivation(map);

    return map;
  }

  @Override
  public Vector backPropagate() {
    Vector previousBlame = new Vector(getBlame().size());

    // Set each prevBlame[i] = b[i] * f'(act[i])
    for (int i = 0; i < 100; i++) {
      double newValue = getBlame().get(i) * Math.cos(getActivation().get(i));
      previousBlame.set(i, newValue);
    }

    // Set prevBlame[100] = b[i] * f'(act[i]), where f'(x) = 1;
    previousBlame.set(100, getBlame().get(100));
    return previousBlame;
  }
}
