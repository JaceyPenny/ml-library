import java.util.HashMap;
import java.util.Map;

public class MaxPooling2DLayer extends Layer {

  private class Tuple3 {
    private int x, y, z;

    public Tuple3(int x, int y, int z) {
      this.x = x;
      this.y = y;
      this.z = z;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }

      if (!(other instanceof Tuple3)) {
        return false;
      }

      Tuple3 otherTuple = (Tuple3) other;
      return otherTuple.x == x && otherTuple.y == y && otherTuple.z == z;
    }

    @Override
    public int hashCode() {
      int hash = 17;
      hash = hash * 37 + Integer.hashCode(x);
      hash = hash * 37 + Integer.hashCode(y);
      hash = hash * 37 + Integer.hashCode(z);

      return hash;
    }
  }

  private int[] inputDimensions;
  private int[] outputDimensions;

  private Map<Tuple3, Integer> tupleToMaxValueMap = new HashMap<>();

  public MaxPooling2DLayer(int[] inputDimensions) {
    super(
        Tensor.countElements(inputDimensions),
        Tensor.countElements(inputDimensions) / 4);

    if (inputDimensions.length != 3) {
      throw new IllegalArgumentException(
          "Your input dimensions must have a dimensionality of 3.");
    }

    if (inputDimensions[0] % 2 != 0 || inputDimensions[1] % 2 != 0) {
      throw new IllegalArgumentException(
          "The first two dimensions of your input must be multiples of two.");
    }

    this.inputDimensions = inputDimensions;
    outputDimensions = new int[inputDimensions.length];
    System.arraycopy(inputDimensions, 0, outputDimensions, 0, inputDimensions.length);

    outputDimensions[0] /= 2;
    outputDimensions[1] /= 2;
  }

  @Override
  public LayerType getLayerType() {
    return LayerType.MAX_POOLING_2D;
  }

  @Override
  public MaxPooling2DLayer copy() {
    return new MaxPooling2DLayer(inputDimensions);
  }

  @Override
  Vector activate(Vector x) {
    Tensor input = new Tensor(x, inputDimensions);
    Tensor activation = new Tensor(getActivation(), outputDimensions);

    for (int k = 0; k < outputDimensions[2]; k++) {
      for (int j = 0; j < outputDimensions[1]; j++) {
        for (int i = 0; i < outputDimensions[0]; i++) {
          double value1 = input.get(i * 2, j * 2, k);
          double value2 = input.get(i * 2 + 1, j * 2, k);
          double value3 = input.get(i * 2, j * 2 + 1, k);
          double value4 = input.get(i * 2 + 1, j * 2 + 1, k);

          double[] values = new double[]{value1, value2, value3, value4};

          int maxIndex = 0;
          for (int index = 1; index < 4; index++) {
            if (values[index] > values[maxIndex]) {
              maxIndex = index;
            }
          }

          tupleToMaxValueMap.put(new Tuple3(i, j, k), maxIndex);
          activation.set(values[maxIndex], i, j, k);
        }
      }
    }

    setActivation(activation);
    return getActivation();
  }

  @Override
  Vector backPropagate() {
    Tensor result = new Tensor(inputDimensions);
    Tensor blame = new Tensor(getBlame(), outputDimensions);

    for (int k = 0; k < outputDimensions[2]; k++) {
      for (int j = 0; j < outputDimensions[1]; j++) {
        for (int i = 0; i < outputDimensions[0]; i++) {
          int maxIndex = tupleToMaxValueMap.get(new Tuple3(i, j, k));

          double value = blame.get(i, j, k);

          switch (maxIndex) {
            case 0:
              result.set(value, i*2, j*2, k);
              break;
            case 1:
              result.set(value, i*2 + 1, j*2, k);
              break;
            case 2:
              result.set(value, i*2, j*2 + 1, k);
              break;
            case 3:
              result.set(value, i*2 + 1, j*2 + 1, k);
              break;
            default:
              break;
          }
        }
      }
    }

    return result;
  }

  @Override
  void resetGradient() {

  }

  @Override
  void updateGradient(Vector x) {

  }

  @Override
  void applyGradient(double learningRate) {

  }

  @Override
  void applyGradient(double learningRate, double momentum) {

  }
}
