public class LayerLinear extends Layer {

    private int inputs;
    private int outputs;

    public LayerLinear(int inputs, int outputs) {
        super(inputs, outputs);

        this.inputs = inputs;
        this.outputs = outputs;
    }

    @Override
    void activate(Vector weights, Vector x) {
        Vector b = new Vector(weights, 0, outputs);
        Vector _MTemp = new Vector(weights, outputs, outputs * inputs);

        Matrix M = Matrix.deserialize(_MTemp, outputs, inputs);

        Matrix _xMatrix = new Matrix(x, Matrix.VectorType.ROW);
        Matrix Mx = Matrix.multiply(_xMatrix, M, false, true);

        Vector productVector = Mx.serialize();

        productVector.add(b);
        super.activation = productVector;
    }

    private void addOuterProductToMatrix(Vector first, Vector second, Matrix target) {
        for (int i = 0; i < first.len; i++) {
            for (int j = 0; j < second.len; j++) {
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

        for (int i = 0; i < averageX.len; i++) {
            averageX.set(i, X.columnMean(i));
        }

        for (int i = 0; i < averageY.len; i++) {
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
        b.addScaled(-1, mx_vector);

        Vector M_serialized = M.serialize();

        weights.set(0, b);
        weights.set(b.len, M_serialized);
    }
}
