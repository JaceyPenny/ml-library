package com.jace.learner;

import com.jace.layer.Layer;
import com.jace.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements SupervisedLearner {
  private List<Layer> layers;

  private double momentum;
  private double learningRate;

  public NeuralNetwork() {
    this.layers = new ArrayList<>();
  }

  public NeuralNetwork copy() {
    final NeuralNetwork newNeuralNetwork = new NeuralNetwork();

    layers.forEach((layer) -> newNeuralNetwork.addLayer(layer.copy()));
    newNeuralNetwork.setMomentum(momentum);
    newNeuralNetwork.setLearningRate(learningRate);

    return newNeuralNetwork;
  }

  public void setMomentum(double momentum) {
    this.momentum = momentum;
  }

  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  public List<Layer> getLayers() {
    return layers;
  }

  @Override
  public String name() {
    return getClass().getSimpleName();
  }

  private void checkLayers() {
    if (layers.size() == 0) {
      throw new IllegalStateException("This network has no layers.");
    }
  }

  public void addLayer(Layer layer) {
    this.layers.add(layer);
  }

  public void initialize() {
    layers.forEach(Layer::initialize);
  }

  public boolean isValid() {
    return layers.size() > 0;
  }

  public boolean isLinearNetwork() {
    return layers.size() == 1 && layers.get(0).getLayerType() == Layer.LayerType.LINEAR;
  }

  @Override
  public Vector predict(Vector in) {
    checkLayers();

    layers.get(0).activate(in);

    for (int i = 1; i < layers.size(); i++) {
      Vector previousActivation = layers.get(i - 1).getActivation();
      layers.get(i).activate(previousActivation);
    }

    return layers.get(layers.size() - 1).getActivation();
  }

  public void updateWeights() {
    for (Layer layer : layers) {
      layer.applyGradient(learningRate, momentum);
    }
  }

  public void backPropagate(Vector target) {
    checkLayers();

    Vector blame = target.copy();
    blame.addScaled(layers.get(layers.size() - 1).getActivation(), -1);
    layers.get(layers.size() - 1).setBlame(blame);

    for (int i = layers.size() - 1; i >= 1; i--) {
      blame = layers.get(i).backPropagate();
      layers.get(i - 1).setBlame(blame);
    }
  }

  public void updateGradient(Vector x) {
    checkLayers();

    Vector previousActivation = x;
    for (Layer layer : layers) {
      layer.updateGradient(previousActivation);
      previousActivation = layer.getActivation();
    }
  }

  public void printTopology() {
    for (int i = 0; i < layers.size(); i++) {
      System.out.printf("%d) %s\n", i, layers.get(i).topologyString());
    }
  }
}
