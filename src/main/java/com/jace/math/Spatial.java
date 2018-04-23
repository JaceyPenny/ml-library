package com.jace.math;

import java.util.function.Function;
import java.util.function.Supplier;

public interface Spatial<T extends Spatial> {
  void fill(double value);

  void fill(Supplier<Double> supplier);

  void addScaled(T other, double scale);

  void scale(double scale);

  void addAll(double value);

  double get(int index);

  void set(int index, double value);

  int size();

  T copy();

  double reduce();

  T map(Function<Double, Double> mapper);
}
