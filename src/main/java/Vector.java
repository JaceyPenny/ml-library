// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.Arrays;

/// Represents a vector of doubles
public class Vector {
  protected double[] vals;
  protected int start;
  private int len;

  public static Vector copy(Vector other) {
    Vector newVector = new Vector(other.len);
    newVector.set(0, other);
    return newVector;
  }

  /// Makes an vector of the specified size
  public Vector(int size) {
    if (size == 0)
      vals = null;
    else
      vals = new double[size];
    start = 0;
    len = size;
  }

  /// Wraps the specified array of doubles
  public Vector(double[] data) {
    vals = data;
    start = 0;
    len = data.length;
  }

  /**
   * This is NOT a copy constructor. It wraps the same buffer of values as v.
   */
  public Vector(Vector v, int begin, int length) {
    vals = v.vals;
    start = v.start + begin;
    len = length;
  }

  /**
   * Unmarshalling constructor
   */
  public Vector(Json n) {
    vals = new double[n.size()];
    for (int i = 0; i < n.size(); i++)
      vals[i] = n.getDouble(i);
    start = 0;
    len = n.size();
  }

  public Json marshal() {
    Json list = Json.newList();
    for (int i = 0; i < len; i++)
      list.add(vals[start + i]);
    return list;
  }

  public int size() {
    return len;
  }

  public double get(int index) {
    return vals[start + index];
  }

  public void set(int index, double value) {
    vals[start + index] = value;
  }

  public void fill(double val) {
    for (int i = 0; i < len; i++)
      vals[start + i] = val;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append('[');
    if (len > 0) {
      sb.append(String.format("%.3f", vals[start]));
      for (int i = 1; i < len; i++) {
        sb.append(",");
        sb.append(String.format("%.3f", vals[start + i]));
      }
    }
    sb.append(']');
    return sb.toString();
  }

  public double squaredMagnitude() {
    double d = 0.0;
    for (int i = 0; i < len; i++)
      d += vals[start + i] * vals[start + i];
    return d;
  }

  public void normalize() {
    double mag = squaredMagnitude();
    if (mag <= 0.0) {
      fill(0.0);
      vals[0] = 1.0;
    } else {
      double s = 1.0 / Math.sqrt(mag);
      for (int i = 0; i < len; i++)
        vals[i] *= s;
    }
  }

  public void add(Vector that) {
    if (that.size() != this.size())
      throw new IllegalArgumentException("mismatching sizes");
    for (int i = 0; i < len; i++)
      vals[start + i] += that.get(i);
  }

  public void set(int startIndex, Vector values) {
    if (startIndex < 0 || startIndex >= len) {
      throw new IllegalArgumentException("startIndex is outside of the Vector bounds");
    }

    int i = startIndex;
    while (i < len && (i - startIndex) < values.len) {
      set(i, values.get(i - startIndex));
      i++;
    }
  }

  public void scale(double scalar) {
    for (int i = 0; i < len; i++)
      vals[start + i] *= scalar;
  }

  public void addScaled(double scalar, Vector that) {
    if (that.size() != this.size())
      throw new IllegalArgumentException("mismatching sizes");
    for (int i = 0; i < len; i++)
      vals[start + i] += scalar * that.get(i);
  }

  public double dotProduct(Vector that) {
    if (that.size() != this.size())
      throw new IllegalArgumentException("mismatching sizes");
    double d = 0.0;
    for (int i = 0; i < len; i++)
      d += get(i) * that.get(i);
    return d;
  }

  public double squaredDistance(Vector that) {
    if (that.size() != this.size())
      throw new IllegalArgumentException("mismatching sizes");
    double d = 0.0;
    for (int i = 0; i < len; i++) {
      double t = get(i) - that.get(i);
      d += (t * t);
    }
    return d;
  }

  public double reduce() {
    double sum = 0;
    for (double d : vals) {
      sum += d;
    }
    return sum;
  }

  public double[] toDoubleArray() {
    double[] array = new double[len];
    System.arraycopy(vals, start, array, 0, len);
    return array;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Vector && Arrays.equals(this.vals, ((Vector) other).vals);
  }
}