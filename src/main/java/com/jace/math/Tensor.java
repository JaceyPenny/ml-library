package com.jace.math;

import java.util.Arrays;

public class Tensor extends Vector {
  private int[] dimensions;
  private int[] dimensionSteps;

  public Tensor(int[] dimensions) {
    super(countElements(dimensions));
    this.dimensions = dimensions;
    initializeDimensionSteps();
  }

  /**
   * General-purpose constructor. Example: <br>
   * com.jace.math.Tensor t(v, new int[] {5, 7, 3});
   */
  public Tensor(Vector values, int[] dimensions) {
    super(values, 0, values.size());
    this.dimensions = new int[dimensions.length];

    int total = countElements(dimensions);
    System.arraycopy(dimensions, 0, this.dimensions, 0, dimensions.length);

    if (total != values.size()) {
      throw new RuntimeException(
          String.format("Mismatching sizes. com.jace.math.Vector has %d, com.jace.math.Tensor has %d", values.size(), total));
    }
    initializeDimensionSteps();
  }

  /**
   * Copy constructor. Copies the dimensions. Wraps the same vector.
   */
  public Tensor(Tensor other) {
    super(other, 0, other.size());
    dimensions = new int[other.dimensions.length];
    System.arraycopy(other.dimensions, 0, dimensions, 0, other.dimensions.length);
    initializeDimensionSteps();
  }

  public int[] getDimensions() {
    return this.dimensions;
  }

  public int getDimension(int dimension) {
    return this.dimensions[dimension];
  }

  public int getLastDimension() {
    return getDimension(dimensions.length - 1);
  }

  private void initializeDimensionSteps() {
    dimensionSteps = new int[dimensions.length];
    dimensionSteps[0] = 1;

    for (int i = 1; i < dimensions.length; i++) {
      dimensionSteps[i] = dimensionSteps[i - 1] * dimensions[i - 1];
    }
  }

  private int calculateIndex(int... position) {
    if (position.length != this.dimensions.length) {
      throw new IllegalArgumentException("Invalid number of dimensions for position");
    }

    int index = 0;

    for (int i = 0; i < dimensions.length; i++) {
      int coordinate = position[i];
      index += coordinate * dimensionSteps[i];
    }

    return index;
  }

  public double get(int... position) {
    int index = calculateIndex(position);
    return super.get(index);
  }

  public void set(double value, int... position) {
    int index = calculateIndex(position);
    super.set(index, value);
  }

  public static Tensor asTensor(Vector value, int[] dimensions) {
    Tensor result;
    if (value instanceof Tensor) {
      result = (Tensor) value;
    } else {
      result = new Tensor(value, dimensions);
    }

    if (!sizesEqual(result.dimensions, dimensions)) {
      if (countElements(result.dimensions) == countElements(dimensions)) {
        result = new Tensor(value, dimensions);
      } else {
        throw new IllegalArgumentException(
            "The desired dimensions are different than the existing com.jace.math.Tensor's dimensions.");
      }
    }

    return result;
  }

  public Tensor[] splitByLastDimension() {
    int numResults = dimensions[dimensions.length - 1];
    Tensor[] result = new Tensor[numResults];
    int lastDimensionStep = dimensionSteps[dimensionSteps.length - 1];

    int[] resultDimensions = new int[dimensions.length - 1];
    System.arraycopy(dimensions, 0, resultDimensions, 0, resultDimensions.length);

    for (int i = 0; i < numResults; i++) {
      Vector segmentVector = new Vector(this, i * lastDimensionStep, lastDimensionStep);
      Tensor segment = new Tensor(segmentVector, resultDimensions);
      result[i] = segment;
    }

    return result;
  }

  private static void checkDimensions(Tensor in, Tensor filter, Tensor out)
      throws IllegalArgumentException {
    int numDimensions = in.dimensions.length;

    if (numDimensions != filter.dimensions.length) {
      throw new IllegalArgumentException("Expected tensors with the same number of dimensions.");
    }

    if (numDimensions != out.dimensions.length) {
      throw new IllegalArgumentException("Expected tensors with the same number of dimensions.");
    }
  }

  public static void convolvePerFilter(Tensor in, Tensor filter, Tensor out) {
    convolvePerFilter(in, filter, out, false);
  }

  static void convolvePerFilter(Tensor in, Tensor filter, Tensor out, boolean flipFilter) {
    convolvePerFilter(in, filter, out, flipFilter, 1);
  }

  static void convolvePerFilter(
      Tensor in, Tensor filter, Tensor out, boolean flipFilter, int stride) {
    checkDimensions(in, filter, out);

    if (in.dimensions[in.dimensions.length - 1] != 1) {
      throw new IllegalArgumentException(
          "Expected the input tensor to be 1 in the last dimension.");
    }

    int numFilters = filter.dimensions[filter.dimensions.length - 1];
    int numOutputs = out.dimensions[out.dimensions.length - 1];

    if (numFilters != numOutputs) {
      throw new IllegalArgumentException(
          "Expected the number of output tensors to be the same as the number of filters.");
    }

    Tensor input = in.splitByLastDimension()[0];
    Tensor[] filters = filter.splitByLastDimension();
    Tensor[] outputs = out.splitByLastDimension();

    for (int i = 0; i < filters.length; i++) {
      convolve(input, filters[i], outputs[i], flipFilter, stride);
    }
  }

  public static void convolvePerLayer(Tensor input, Tensor filter, Tensor result) {
    convolvePerLayer(input, filter, result, false);
  }

  public static void convolvePerLayer(Tensor input, Tensor filter, Tensor result, boolean flipFilter) {
    convolvePerLayer(input, filter, result, flipFilter, 1);
  }

  public static void convolvePerLayer(Tensor input, Tensor filter, Tensor result, boolean flipFilter, int stride) {
    if (input.getLastDimension() != filter.getLastDimension()) {
      throw new IllegalArgumentException("Expected the input and filter to have the same last dimension.");
    }

    if (result.getLastDimension() != 1) {
      throw new IllegalArgumentException("Expected the result com.jace.math.Tensor to have a final dimension of 1.");
    }

    Tensor[] inputSplit = input.splitByLastDimension();
    Tensor[] filterSplit = filter.splitByLastDimension();
    Tensor resultLowered = result.splitByLastDimension()[0];

    for (int i = 0; i < inputSplit.length; i++) {
      convolve(inputSplit[i], filterSplit[i], resultLowered, flipFilter, stride);
    }
  }

  public static void convolve(Tensor in, Tensor filter, Tensor out) {
    convolve(in, filter, out, false);
  }

  static void convolve(Tensor in, Tensor filter, Tensor out, boolean flipFilter) {
    convolve(in, filter, out, flipFilter, 1);
  }

  /**
   * The result is added to the existing contents of out. It does not replace the existing contents
   * of out. Padding is computed as necessary to fill the out tensor.
   *
   * @param in         the input com.jace.math.Tensor
   * @param filter     the filter to convolve within
   * @param out        the target tensor to add the result to.
   * @param flipFilter whether or not to flip the filter in all dimensions
   * @param stride     {unsure}
   */
  static void convolve(Tensor in, Tensor filter, Tensor out, boolean flipFilter, int stride) {
    // Pre-compute some values
    checkDimensions(in, filter, out);

    int numDimensions = in.dimensions.length;

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

          if (innerK[i] < filter.dimensions[i] && outerK[i] + innerK[i] - padding < in.dimensions[i]) {
            break;
          }

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

  public static boolean sizesEqual(Tensor first, Tensor second) {
    return sizesEqual(first.dimensions, second.dimensions);
  }

  private static boolean sizesEqual(int[] firstDimensions, int[] secondDimensions) {
    if (firstDimensions.length != secondDimensions.length) {
      return false;
    }

    for (int i = 0; i < firstDimensions.length; i++) {
      if (firstDimensions[i] != secondDimensions[i]) {
        return false;
      }
    }

    return true;
  }

  @Override
  public String toString() {
    if (dimensions.length == 1) {
      return super.toString();
    }

    StringBuilder stringBuilder = new StringBuilder();
    Tensor[] split = splitByLastDimension();
    for (Tensor tensor : split) {
      stringBuilder.append(tensor.toString());
      stringBuilder.append('\n');
    }

//    stringBuilder.append('\n');
    return stringBuilder.toString();
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
