package com.jace.util;

import com.jace.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;

public class Strings {

  public static <T> String join(T[] strings, String joinString) {
    return join(Arrays.asList(strings), joinString);
  }

  public static <T> String join(List<T> strings, String joinString) {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < strings.size() - 1; i++) {
      builder.append(strings.get(i)).append(joinString);
    }
    builder.append(strings.get(strings.size() - 1));
    return builder.toString();
  }

  public static <T> String join(T[] strings, char joinChar) {
    return join(strings, "" + joinChar);
  }

  public static <T> String join(List<T> strings, char joinChar) {
    return join(strings, "" + joinChar);
  }

  public static String join(Vector vector, String joinString) {
    Double[] doubles = DoubleStream.of(vector.toDoubleArray()).boxed().toArray(Double[]::new);
    return join(doubles, joinString);
  }

  public static String join(Vector vector, char joinChar) {
    return join(vector, "" + joinChar);
  }
}
