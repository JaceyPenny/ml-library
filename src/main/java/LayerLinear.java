public class LayerLinear extends Layer {

  public LayerLinear(int inputs, int outputs) {
    super(inputs, outputs);
  }

  @Override
  void activate(Vector weights, Vector x) {
    Vector b = WeightUtils.extract_b(weights, getOutputs());
    Matrix M = WeightUtils.extract_M(weights, getInputs(), getOutputs());

    Matrix _xMatrix = new Matrix(x, Matrix.VectorType.ROW);
    Matrix Mx = Matrix.multiply(_xMatrix, M, false, true);

    Vector productVector = Mx.serialize();

    productVector.add(b);
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

  public void ordinaryLeastSquares(Matrix X, Matrix Y, Vector weights) {
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
      y_i_minus_average_y.addScaled(-1, averageY);

      Vector x_i_minus_average_x = Vector.copy(X.row(i));
      x_i_minus_average_x.addScaled(-1, averageX);

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
    b.addScaled(-1, mx_vector);

    WeightUtils.set_b(weights, b);
    WeightUtils.set_M(weights, M);
  }

  @Override
  void backPropagate(Vector weights, Vector previousBlame) {
    Matrix M = WeightUtils.extract_M(weights, getInputs(), getOutputs());
    Matrix blameMatrix = new Matrix(previousBlame, Matrix.VectorType.COLUMN);

    Matrix product = Matrix.multiply(M, blameMatrix, true, false);
    setBlame(product.serialize());
  }

  @Override
  void updateGradient(Vector x, Vector gradient) {
    Vector b = WeightUtils.extract_b(gradient, getOutputs());
    Matrix M = WeightUtils.extract_M(gradient, getInputs(), getOutputs());

    b.add(getBlame());

    Matrix outerProduct = Vector.outerProduct(getBlame(), x);
    M.addScaled(M, 1);

    WeightUtils.set_b(gradient, b);
    WeightUtils.set_M(gradient, M);
  }
}
