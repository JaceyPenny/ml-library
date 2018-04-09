// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.Arrays;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Represents a vector of doubles
 */
public class Vector implements Spatial<Vector> {
  protected double[] values;
  protected int startIndex;
  private int length;

  public static Vector copy(Vector other) {
    Vector newVector = new Vector(other.length);
    newVector.set(0, other);
    return newVector;
  }

  /**
   * Makes an vector of the specified size
   */
  public Vector(int size) {
    values = new double[size];
    startIndex = 0;
    length = size;
  }

  /**
   * Wraps the specified array of doubles
   */
  public Vector(double[] data) {
    values = data;
    startIndex = 0;
    length = data.length;
  }

  /**
   * This is NOT a copy constructor. It wraps the same buffer of values as other.
   */
  public Vector(Vector other, int begin, int length) {
    values = other.values;
    startIndex = other.startIndex + begin;
    this.length = length;
  }

  /**
   * Unmarshalling constructor
   */
  public Vector(Json json) {
    values = new double[json.size()];
    for (int i = 0; i < json.size(); i++) {
      values[i] = json.getDouble(i);
    }
    startIndex = 0;
    length = json.size();
  }

  public Json marshal() {
    Json list = Json.newList();
    for (int i = 0; i < size(); i++) {
      list.add(get(i));
    }
    return list;
  }

  public Vector map(Function<Double, Double> mapper) {
    Vector newVector = new Vector(size());
    for (int i = 0; i < size(); i++) {
      newVector.set(i, mapper.apply(get(i)));
    }
    return newVector;
  }

  public Vector copy() {
    return Vector.copy(this);
  }

  public int size() {
    return length;
  }

  public double get(int index) {
    return values[startIndex + index];
  }

  public void set(int index, double value) {
    values[startIndex + index] = value;
  }

  public void fill(Supplier<Double> supplier) {
    for (int i = 0; i < size(); i++) {
      set(i, supplier.get());
    }
  }

  public void fill(double value) {
    for (int i = 0; i < length; i++) {
      set(i, value);
    }
  }

  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    stringBuilder.append('[');
    if (length > 0) {
      stringBuilder.append(String.format("%6.5f", values[startIndex]));
      for (int i = 1; i < length; i++) {
        stringBuilder.append(", ");
        stringBuilder.append(String.format("%6.5f", values[startIndex + i]));
      }
    }
    stringBuilder.append(']');
    return stringBuilder.toString();
  }

  public double squaredMagnitude() {
    double result = 0.0;
    for (int i = 0; i < length; i++) {
      result += values[startIndex + i] * values[startIndex + i];
    }
    return result;
  }

  public void normalize() {
    double magnitude = squaredMagnitude();
    if (magnitude <= 0.0) {
      fill(0.0);
      values[0] = 1.0;
    } else {
      double ratio = 1.0 / Math.sqrt(magnitude);
      for (int i = 0; i < length; i++) {
        values[i] *= ratio;
      }
    }
  }

  public int maxIndex() {
    int maxIndex = 0;
    double maxValue = get(0);
    for (int i = 0; i < size(); i++) {
      if (get(i) > maxValue) {
        maxIndex = i;
        maxValue = get(i);
      }
    }
    return maxIndex;
  }

  public void add(Vector that) {
    if (that.size() != this.size()) {
      throw new IllegalArgumentException("mismatching sizes");
    }

    for (int i = 0; i < length; i++) {
      values[startIndex + i] += that.get(i);
    }
  }

  public void set(int startIndex, Vector values) {
    if (startIndex < 0 || startIndex >= length) {
      throw new IllegalArgumentException("startIndex is outside of the Vector bounds");
    }

    int i = startIndex;
    while (i < length && (i - startIndex) < values.length) {
      set(i, values.get(i - startIndex));
      i++;
    }
  }

  public void scale(double scalar) {
    for (int i = 0; i < length; i++) {
      values[startIndex + i] *= scalar;
    }
  }

  public void addScaled(Vector that, double scalar) {
    if (that.size() != this.size()) {
      throw new IllegalArgumentException("mismatching sizes");
    }

    for (int i = 0; i < length; i++) {
      values[startIndex + i] += scalar * that.get(i);
    }
  }

  public void addAll(double scalar) {
    for (int i = 0; i < length; i++) {
      set(i, get(i) + scalar);
    }
  }

  public double dotProduct(Vector that) {
    if (that.size() != this.size()) {
      throw new IllegalArgumentException("mismatching sizes");
    }

    double result = 0.0;
    for (int i = 0; i < length; i++) {
      result += get(i) * that.get(i);
    }
    return result;
  }

  public double squaredDistance(Vector that) {
    if (that.size() != this.size()) {
      throw new IllegalArgumentException("mismatching sizes");
    }

    double result = 0.0;
    for (int i = 0; i < length; i++) {
      double distance = get(i) - that.get(i);
      result += distance * distance;
    }
    return result;
  }

  public double reduce() {
    double sum = 0;
    for (int i = 0; i < size(); i++) {
      sum += get(i);
    }
    return sum;
  }

  public double[] toDoubleArray() {
    double[] array = new double[length];
    System.arraycopy(values, startIndex, array, 0, length);
    return array;
  }

  public static Matrix outerProduct(Vector a, Vector b) {
    if (a.size() == 0 || b.size() == 0) {
      throw new IllegalArgumentException("Cannot compute the outer product for empty vectors.");
    }

    Matrix result = new Matrix(a.size(), b.size());
    for (int i = 0; i < a.size(); i++) {
      for (int j = 0; j < b.size(); j++) {
        result.set(i, j, a.get(i) * b.get(j));
      }
    }

    return result;
  }

  public Matrix asMatrix(Matrix.VectorType vectorType) {
    return new Matrix(this, vectorType);
  }

  public double errorAgainst(Vector other) {
    if (size() != other.size()) {
      throw new IllegalArgumentException("These Vectors are not the same size");
    }

    double error = 0.0;
    for (int i = 0; i < size(); i++) {
      error += Math.abs(get(i) - other.get(i));
    }

    return error;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Vector && Arrays.equals(this.values, ((Vector) other).values);
  }
}