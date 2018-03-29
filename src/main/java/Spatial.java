import java.util.function.Supplier;

public interface Spatial<T extends Spatial> {
  void fill(double value);
  void fill(Supplier<Double> supplier);
  void addScaled(T other, double scale);
  void scale(double scale);
}
