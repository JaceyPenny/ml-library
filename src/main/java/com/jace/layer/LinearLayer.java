package com.jace.layer;

import com.jace.Main;
import com.jace.math.Matrix;
import com.jace.math.Vector;

public class LinearLayer extends ConnectedLayer<Matrix, Vector> {

  public LinearLayer(int inputs, int outputs) {
    super(inputs, outputs);
  }

  @Override
  public Layer.LayerType getLayerType() {
    return Layer.LayerType.LINEAR;
  }

  @Override
  public void initialize() {
    super.initialize();

    setWeights(new Matrix(getOutputs(), getInputs()));
    setBias(new Vector(getOutputs()));

    fillAll(() -> Math.max(0, 1.0 / getInputs()) * Main.RANDOM.nextGaussian());
  }

  @Override
  public LinearLayer copy() {
    return new LinearLayer(getInputs(), getOutputs());
  }

  @Override
  protected void resetGradient() {
    if (getWeightsGradient() == null) {
      setWeightsGradient(new Matrix(getOutputs(), getInputs()));
    }

    if (getBiasGradient() == null) {
      setBiasGradient(new Vector(getOutputs()));
    }

    super.resetGradient();
  }

  @Override
  public Vector activate(Vector x) {
    Matrix xMatrix = x.asMatrix(Matrix.VectorType.COLUMN);
    Matrix Mx = Matrix.multiply(getWeights(), xMatrix);

    Vector productVector = Mx.serialize();

    productVector.add(getBias());
    setActivation(productVector);
    return getActivation();
  }

  private void addOuterProductToMatrix(Vector first, Vector second, Matrix target) {
    for (int i = 0; i < first.size(); i++) {
      for (int j = 0; j < second.size(); j++) {
        double first_i = first.get(i);
        double second_j = second.get(j);
        double currentValue = target.get(i, j);
        target.set(i, j, currentValue + first_i * second_j);
      }
    }
  }

  public void ordinaryLeastSquares(Matrix X, Matrix Y) {
    Vector averageX = new Vector(X.cols());
    Vector averageY = new Vector(Y.cols());

    for (int i = 0; i < averageX.size(); i++) {
      averageX.set(i, X.columnMean(i));
    }

    for (int i = 0; i < averageY.size(); i++) {
      averageY.set(i, Y.columnMean(i));
    }

    Matrix firstTerm = new Matrix(Y.cols(), X.cols());
    Matrix secondTerm = new Matrix(X.cols(), X.cols());

    for (int i = 0; i < X.rows(); i++) {
      Vector y_i_minus_average_y = Vector.copy(Y.row(i));
      y_i_minus_average_y.addScaled(averageY, -1);

      Vector x_i_minus_average_x = Vector.copy(X.row(i));
      x_i_minus_average_x.addScaled(averageX, -1);

      addOuterProductToMatrix(y_i_minus_average_y, x_i_minus_average_x, firstTerm);
      addOuterProductToMatrix(x_i_minus_average_x, x_i_minus_average_x, secondTerm);
    }

    secondTerm = secondTerm.pseudoInverse();

    Matrix M = Matrix.multiply(firstTerm, secondTerm);

    Matrix averageX_matrix = new Matrix(averageX, Matrix.VectorType.COLUMN);

    Matrix Mx_product = Matrix.multiply(M, averageX_matrix);
    Vector mx_vector = Mx_product.serialize();

    Vector b = Vector.copy(averageY);

    // b -= mx_vector
    b.addScaled(mx_vector, -1);

    setWeights(M);
    setBias(b);
  }

  @Override
  public Vector backPropagate() {
    Matrix blameMatrix = getBlame().asMatrix(Matrix.VectorType.COLUMN);

    Matrix previousBlameMatrix = Matrix.multiply(getWeights(), blameMatrix, true, false);
    return previousBlameMatrix.serialize();
  }

  @Override
  public void updateGradient(Vector x) {
    addOuterProductToMatrix(getBlame(), x, getWeightsGradient());

    switch (getRegularizationType()) {
      case L1:
        for (int i = 0; i < getWeights().size() - 1; i++) {
          double sign = -1 * Math.signum(getWeights().get(i));
          getWeightsGradient().set(i, getWeightsGradient().get(i) + sign * getRegularizationAmount());
        }
        break;
      case L2:
        for (int i = 0; i < getWeights().size() - 1; i++) {
          double shift = getWeights().get(i) * getRegularizationAmount();
          getWeightsGradient().set(i, getWeightsGradient().get(i) - shift);
        }
        break;
      case NONE:
      default:
        break;
    }

    getBiasGradient().add(getBlame());
  }
}
