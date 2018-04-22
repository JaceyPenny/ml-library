package com.jace.evaluator;

import com.jace.layer.ConnectedLayer;
import com.jace.layer.Layer;
import com.jace.learner.NeuralNetwork;
import com.jace.math.Matrix;
import com.jace.math.Spatial;
import com.jace.math.Vector;

import java.util.HashMap;
import java.util.Map;

public class GradientEvaluator extends LearnerEvaluator<NeuralNetwork> {
  private static final double DELTA = 0.0001;

  private Matrix testFeatures;
  private Matrix testLabels;

  public GradientEvaluator(NeuralNetwork learner) {
    super(learner);
  }

  public void setTestData(Matrix testFeatures, Matrix testLabels) {
    this.testFeatures = testFeatures;
    this.testLabels = testLabels;
  }

  public double computeSumSquaredError() {
    return super.computeSumSquaredError(testFeatures, testLabels);
  }

  public void checkAgainstFiniteDifferencing() {
    Map<ConnectedLayer, Spatial> layerToWeightGradientsMap = new HashMap<>();
    Map<ConnectedLayer, Spatial> layerToBiasGradientsMap = new HashMap<>();

    for (Layer layer : getLearner().getLayers()) {
      if (layer instanceof ConnectedLayer) {
        ConnectedLayer connectedLayer = (ConnectedLayer) layer;

        Spatial weightsGradient = computeWeightsGradient(connectedLayer);
        Spatial biasGradient = computeBiasGradient(connectedLayer);
        layerToWeightGradientsMap.put(connectedLayer, weightsGradient);
        layerToBiasGradientsMap.put(connectedLayer, biasGradient);
      }
    }

    calculateEmpiricalGradient();

    int currentLayer = 0;
    for (Layer layer : getLearner().getLayers()) {
      if (layer instanceof ConnectedLayer) {
        ConnectedLayer connectedLayer = (ConnectedLayer) layer;

        Spatial calculatedWeightsGradient = layerToWeightGradientsMap.get(connectedLayer);
        Spatial calculatedBiasGradient = layerToBiasGradientsMap.get(connectedLayer);

        Spatial empiricalWeightsGradient = connectedLayer.getWeightsGradient();
        Spatial empiricalBiasGradient = connectedLayer.getBiasGradient();

        System.out.println("============================");
        System.out.printf("LAYER %d: %s\n\n", currentLayer, connectedLayer.getLayerType());
        System.out.println("Weights (finite differencing):");
        System.out.println(calculatedWeightsGradient.toString().trim());
        System.out.println("\nWeights (empirical):");
        System.out.println(empiricalWeightsGradient.toString().trim());
        System.out.println("\nSimilarity... " + compareVectors(calculatedWeightsGradient, empiricalWeightsGradient));
        System.out.println();

        System.out.println("\nBiases (finite differencing):");
        System.out.println(calculatedBiasGradient.toString().trim());
        System.out.println("\nBiases (empirical):");
        System.out.println(empiricalBiasGradient.toString().trim());
        System.out.println("\nSimilarity... " + compareVectors(calculatedBiasGradient, empiricalBiasGradient));
        System.out.println("============================\n\n");
      } else {
        System.out.println("============================");
        System.out.printf("LAYER %d: %s (no weights, skipping)\n", currentLayer, layer.getLayerType());
        System.out.println("============================\n\n");
      }

      currentLayer++;
    }
  }

  private String compareVectors(Spatial first, Spatial second) {
    int locations = 0;
    for (int i = 0; i < first.size(); i++) {
      if (Math.abs(first.get(i) - second.get(i)) > 1e-4) {
        locations++;
      }
    }

    if (locations > 0) {
      double totalDifference = first.reduce() - second.reduce();
      return String.format(
          "DIFFERENT (in %d location%s, total difference: %.4f, average difference: %.4f)",
          locations,
          (locations == 1) ? "" : "s",
          totalDifference,
          totalDifference / locations);
    } else {
      return "SAME";
    }
  }

  private Spatial computeWeightsGradient(ConnectedLayer connectedLayer) {
    Spatial weightsGradient = connectedLayer.getWeightsGradient().copy();
    weightsGradient.fill(0);

    for (int i = 0; i < connectedLayer.getWeights().size(); i++) {
      double weight = connectedLayer.getWeights().get(i);

      connectedLayer.getWeights().set(i, weight - DELTA);
      double lowerSumSquaredError = computeSumSquaredError();

      connectedLayer.getWeights().set(i, weight + DELTA);
      double upperSumSquaredError = computeSumSquaredError();

      connectedLayer.getWeights().set(i, weight);

      double gradient = (lowerSumSquaredError - upperSumSquaredError) / (4 * DELTA);
      weightsGradient.set(i, gradient);
    }

    return weightsGradient;
  }

  private Spatial computeBiasGradient(ConnectedLayer connectedLayer) {
    Spatial biasGradient = connectedLayer.getBiasGradient().copy();
    biasGradient.fill(0);

    for (int i = 0; i < connectedLayer.getBias().size(); i++) {
      double bias = connectedLayer.getBias().get(i);

      connectedLayer.getBias().set(i, bias - DELTA);
      double lowerSumSquaredError = computeSumSquaredError();

      connectedLayer.getBias().set(i, bias + DELTA);
      double upperSumSquaredError = computeSumSquaredError();

      connectedLayer.getBias().set(i, bias);

      double gradient = (lowerSumSquaredError - upperSumSquaredError) / (4 * DELTA);
      biasGradient.set(i, gradient);
    }

    return biasGradient;
  }

  private void calculateEmpiricalGradient() {
    for (int i = 0; i < testFeatures.rows(); i++) {
      Vector input = testFeatures.row(i);
      Vector output = testLabels.row(i);

      getLearner().predict(input);
      getLearner().backPropagate(output);
      getLearner().updateGradient(input);
    }
  }
}
