public class LayerLinear extends Layer {
  private Matrix weights;
  private Vector bias;

  private Matrix weightsGradient;
  private Vector biasGradient;

  public LayerLinear(int inputs, int outputs) {
    super(inputs, outputs);

    weights = new Matrix(outputs, inputs);
    bias = new Vector(outputs);

    resetGradient();
  }

  @Override
  protected void resetGradient() {
    if (weightsGradient == null) {
      weightsGradient = new Matrix(getOutputs(), getInputs());
    }

    if (bias == null) {
      biasGradient = new Vector(getOutputs());
    }

    weightsGradient.fill(0);
    bias.fill(0);
  }

  @Override
  void activate(Vector x) {
    Matrix _xMatrix = new Matrix(x, Matrix.VectorType.ROW);
    Matrix Mx = Matrix.multiply(_xMatrix, weights, false, true);

    Vector productVector = Mx.serialize();

    productVector.add(bias);
    setActivation(productVector);
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

    weights = M;
    bias = b;
  }

  @Override
  void backPropagate(Vector previousBlame) {
    Matrix blameMatrix = previousBlame.asMatrix(Matrix.VectorType.COLUMN);

    Matrix product = Matrix.multiply(weights, blameMatrix, true, false);
    setBlame(product.serialize());
  }

  @Override
  void updateGradient(Vector x) {
    Matrix outerProduct = Vector.outerProduct(getBlame(), x);
    weightsGradient.addScaled(outerProduct, 1);

    biasGradient.add(getBlame());
  }

  @Override
  void applyGradient(double learningRate) {
    weights.addScaled(weightsGradient, learningRate);
    bias.addScaled(biasGradient, learningRate);

    resetGradient();
  }
}
