import java.util.function.Supplier;

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
  public LayerType getLayerType() {
    return LayerType.LINEAR;
  }

  public void setWeights(Matrix weights) {
    this.weights = weights;
  }

  public Matrix getWeights() {
    return weights;
  }

  public void setBias(Vector bias) {
    this.bias = bias;
  }

  public Vector getBias() {
    return bias;
  }

  public void initializeWeights() {
    fillAll(() -> Math.max(0.03, 1.0 / getInputs()) * NeuralNetwork.RANDOM.nextGaussian());
  }

  private void fillAll(Supplier<Double> supplier) {
    weights.fill(supplier);
    bias.fill(supplier);
  }

  @Override
  protected void resetGradient() {
    if (weightsGradient == null) {
      weightsGradient = new Matrix(getOutputs(), getInputs());
    }

    if (biasGradient == null) {
      biasGradient = new Vector(getOutputs());
    }

    weightsGradient.fill(0);
    biasGradient.fill(0);
  }

  @Override
  Vector activate(Vector x) {
    Matrix xMatrix = x.asMatrix(Matrix.VectorType.COLUMN);
    Matrix Mx = Matrix.multiply(weights, xMatrix);

    Vector productVector = Mx.serialize();

    productVector.add(bias);
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

    weights = M;
    bias = b;
  }

  @Override
  protected Vector backPropagate() {
    Matrix blameMatrix = getBlame().asMatrix(Matrix.VectorType.COLUMN);

    Matrix previousBlameMatrix = Matrix.multiply(weights, blameMatrix, true, false);
    return previousBlameMatrix.serialize();
  }

  @Override
  void updateGradient(Vector x) {
    addOuterProductToMatrix(getBlame(), x, weightsGradient);
    biasGradient.add(getBlame());
  }

  @Override
  void applyGradient(double learningRate) {
    applyGradient(learningRate, 0);
  }

  @Override
  void applyGradient(double learningRate, double momentum) {
    weights.addScaled(weightsGradient, learningRate);
    bias.addScaled(biasGradient, learningRate);

    weightsGradient.scale(momentum);
    biasGradient.scale(momentum);
  }
}
