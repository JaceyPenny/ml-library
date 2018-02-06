public class WeightUtils {

  public static Vector extract_b(Vector weights, int outputs) {
    return new Vector(weights, 0, outputs);
  }

  public static Matrix extract_M(Vector weights, int inputs, int outputs) {
    Vector _MTemp = new Vector(weights, outputs, outputs * inputs);
    return Matrix.deserialize(_MTemp, outputs, inputs);
  }

  public static void set_b(Vector weights, Vector b) {
    weights.set(0, b);
  }

  public static void set_M(Vector weights, Matrix M) {
    Vector M_vector = M.serialize();
    weights.set(M.rows(), M_vector);
  }
}
