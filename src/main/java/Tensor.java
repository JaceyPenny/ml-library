import java.util.Arrays;

public class Tensor extends Vector {
  private int[] dimensions;

  /**
   * General-purpose constructor. Example: <br>
   * Tensor t(v, new int[] {5, 7, 3});
   */
  public Tensor(Vector values, int[] dimensions) {
    super(values, 0, values.size());
    this.dimensions = new int[dimensions.length];

    int total = 1;

    for (int i = 0; i < dimensions.length; i++) {
      this.dimensions[i] = dimensions[i];
      total *= dimensions[i];
    }

    if (total != values.size()) {
      throw new RuntimeException(
          String.format("Mismatching sizes. Vector has %d, Tensor has %d", values.size(), total));
    }
  }

  /**
   * Copy constructor. Copies the dimensions. Wraps the same vector.
   */
  public Tensor(Tensor other) {
    super(other, 0, other.size());
    dimensions = new int[other.dimensions.length];
    System.arraycopy(other.dimensions, 0, dimensions, 0, other.dimensions.length);
  }

  /**
   * The result is added to the existing contents of out. It does not replace the existing contents
   * of out. Padding is computed as necessary to fill the out tensor.
   *
   * @param in         the input Tensor
   * @param filter     the filter to convolve within
   * @param out        the target tensor to add the result to.
   * @param flipFilter whether or not to flip the filter in all dimensions
   * @param stride     {unsure}
   */
  static void convolve(Tensor in, Tensor filter, Tensor out, boolean flipFilter, int stride) {
    // Pre-compute some values
    int numDimensions = in.dimensions.length;

    if (numDimensions != filter.dimensions.length) {
      throw new RuntimeException("Expected tensors with the same number of dimensions");
    }

    if (numDimensions != out.dimensions.length) {
      throw new RuntimeException("Expected tensors with the same number of dimensions");
    }

    int[] innerK = new int[numDimensions];
    int[] outerK = new int[numDimensions];
    int[] innerStep = new int[numDimensions];
    int[] filterStep = new int[numDimensions];
    int[] outerStep = new int[numDimensions];

    // Compute step sizes
    innerStep[0] = 1;
    filterStep[0] = 1;
    outerStep[0] = 1;

    for (int i = 1; i < numDimensions; i++) {
      innerStep[i] = innerStep[i - 1] * in.dimensions[i - 1];
      filterStep[i] = filterStep[i - 1] * filter.dimensions[i - 1];
      outerStep[i] = outerStep[i - 1] * out.dimensions[i - 1];
    }

    int filterTail = filterStep[numDimensions - 1] * filter.dimensions[numDimensions - 1] - 1;

    // Do convolution
    int op = 0;
    int ip = 0;
    int fp = 0;
    for (int i = 0; i < numDimensions; i++) {
      outerK[i] = 0;
      innerK[i] = 0;
      int padding = (stride * (out.dimensions[i] - 1) + filter.dimensions[i] - in.dimensions[i]) / 2;
      int adj = (padding - Math.min(padding, outerK[i])) - innerK[i];
      innerK[i] += adj;
      fp += adj * filterStep[i];
    }

    // outerK
    while (true) {
      double val = 0.0;

      // Fix up the initial innerK positions
      for (int i = 0; i < numDimensions; i++) {
        int padding = (stride * (out.dimensions[i] - 1) + filter.dimensions[i] - in.dimensions[i]) / 2;
        int adj = (padding - Math.min(padding, outerK[i])) - innerK[i];
        innerK[i] += adj;
        fp += adj * filterStep[i];
        ip += adj * innerStep[i];
      }

      // innerK
      while (true) {
        val += (in.get(ip) * filter.get(flipFilter ? filterTail - fp : fp));

        // increment the innerK position
        int i;
        for (i = 0; i < numDimensions; i++) {
          innerK[i]++;
          ip += innerStep[i];
          fp += filterStep[i];
          int padding = (stride * (out.dimensions[i] - 1) + filter.dimensions[i] - in.dimensions[i]) / 2;
          if (innerK[i] < filter.dimensions[i] && outerK[i] + innerK[i] - padding < in.dimensions[i])
            break;
          int adj = (padding - Math.min(padding, outerK[i])) - innerK[i];
          innerK[i] += adj;
          fp += adj * filterStep[i];
          ip += adj * innerStep[i];
        }

        if (i >= numDimensions) {
          break;
        }
      }

      out.set(op, out.get(op) + val);

      // increment the outerK position
      int i;
      for (i = 0; i < numDimensions; i++) {
        outerK[i]++;

        op += outerStep[i];
        ip += stride * innerStep[i];

        if (outerK[i] < out.dimensions[i]) {
          break;
        }

        op -= outerK[i] * outerStep[i];
        ip -= outerK[i] * stride * innerStep[i];

        outerK[i] = 0;
      }

      if (i >= numDimensions) {
        break;
      }
    }
  }

  public static int countElements(int[] dimensions) {
    return Arrays.stream(dimensions).reduce(1, (id, next) -> id * next);
  }

  /**
   * Throws an exception if something is wrong.
   */
  public static void test() {
    test1D();
    test2D();
  }

  private static void test1D() {
    Vector in = new Vector(new double[]{2, 3, 1, 0, 1});
    Tensor tin = new Tensor(in, new int[]{5});

    Vector k = new Vector(new double[]{1, 0, 2});
    Tensor tk = new Tensor(k, new int[]{3});

    Vector out = new Vector(7);
    Tensor tout = new Tensor(out, new int[]{7});

    Tensor.convolve(tin, tk, tout, true, 1);

    //     2 3 1 0 1
    // 2 0 1 --->
    Vector expected = new Vector(new double[]{2, 3, 5, 6, 3, 0, 2});
    if (Math.sqrt(out.squaredDistance(expected)) > 1e-10) {
      throw new RuntimeException("wrong");
    }
  }

  private static void test2D() {
    Vector in = new Vector(new double[]{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    Tensor tin = new Tensor(in, new int[]{3, 3});

    Vector k = new Vector(new double[]{
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    });

    Tensor tk = new Tensor(k, new int[]{3, 3});

    Vector out = new Vector(9);
    Tensor tout = new Tensor(out, new int[]{3, 3});

    Tensor.convolve(tin, tk, tout, false, 1);

    Vector expected = new Vector(new double[]{
        -13, -20, -17,
        -18, -24, -18,
        13, 20, 17
    });

    if (Math.sqrt(out.squaredDistance(expected)) > 1e-10) {
      throw new RuntimeException("wrong");
    }
  }
}
