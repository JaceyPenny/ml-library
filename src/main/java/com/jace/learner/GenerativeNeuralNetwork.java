package com.jace.learner;

import com.jace.Main;
import com.jace.layer.Layer;
import com.jace.layer.LinearLayer;
import com.jace.math.Matrix;
import com.jace.math.Vector;
import com.jace.util.Console;

public class GenerativeNeuralNetwork extends NeuralNetwork {
  private static int MAX_REPS = 10000000;

  private int trainingRows;

  private Matrix estimatedState;

  private Vector gradient;

  private int width;
  private int height;

  private int degreesOfFreedom;

  private int trainingRow = 0;


  public GenerativeNeuralNetwork(int width, int height, int degreesOfFreedom, int trainingRows) {
    super();

    this.width = width;
    this.height = height;
    this.degreesOfFreedom = degreesOfFreedom;
    this.trainingRows = trainingRows;
  }

  public void setEstimatedState(Matrix estimatedState) {
    this.estimatedState = estimatedState;
  }

  public Matrix getEstimatedState() {
    return estimatedState;
  }

  @Override
  public void addLayer(Layer layer) {
    if (getLayers().isEmpty()) {
      estimatedState = new Matrix(trainingRows, layer.getInputs() - 2);
      gradient = new Vector(layer.getInputs());
    }

    super.addLayer(layer);
  }

  public void trainUnsupervised(Matrix observationMatrix) {
    estimatedState.fill(0);

    int channels =  observationMatrix.cols() / (width * height);
    Vector feature = new Vector(2 + degreesOfFreedom);
    Vector label = new Vector(channels);

    for (int j = 0; j < 10; j++) {
      for (int i = 0; i < 10000000; i++) {
        if (i % 1000 == 0) {
          Console.progress("Training epoch " + j, (double) i / MAX_REPS * 100);
        }

        trainingRow = Main.RANDOM.nextInt(observationMatrix.rows());

        int p = Main.RANDOM.nextInt(width);
        int q = Main.RANDOM.nextInt(height);

        Vector v_feature = estimatedState.row(trainingRow);
        feature.set(0, p / (double) width);
        feature.set(1, q / (double) height);

        for (int l = 0; l < degreesOfFreedom; l++) {
          feature.set(2 + l, v_feature.get(l));
        }

        Vector observationRow = observationMatrix.row(trainingRow);
        int s = channels * (width * q + p);
        for (int l = 0; l < channels; l++) {
          label.set(l, observationRow.get(s + l));
        }

        predict(feature);
        backPropagate(label);
        updateGradient(feature);
        updateWeights();
      }

      setLearningRate(getLearningRate() * 0.75);
    }
  }

  @Override
  public void updateWeights() {
    super.updateWeights();

    Vector stateGradient = new Vector(gradient, 2, 2);

    estimatedState.row(trainingRow).addScaled(stateGradient, getLearningRate());
    gradient.scale(getMomentum());
  }

  @Override
  public void updateGradient(Vector x) {
    super.updateGradient(x);

    if (!(getLayers().get(0) instanceof LinearLayer)) {
      throw new IllegalStateException("Generative neural networks must have a LinearLayer for the first layer.");
    }

    LinearLayer firstLayer = (LinearLayer) getLayers().get(0);

    Matrix weights = firstLayer.getWeights();
    Matrix blameMatrix = firstLayer.getBlame().asMatrix(Matrix.VectorType.COLUMN);

    Matrix gradientMatrix = Matrix.multiply(weights, blameMatrix, true, false);
    gradient.addScaled(gradientMatrix.serialize(), -1);
  }
}
