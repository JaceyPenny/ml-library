import com.jace.math.Tensor;
import com.jace.math.Vector;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.assertEquals;

@RunWith(JUnit4.class)
public class TensorTest extends BaseTest {

  @Test
  public void testCountElements_small() {
    int[] dimensions = new int[]{1, 2, 3};
    int actualElements = Tensor.countElements(dimensions);
    int expectedElements = 1 * 2 * 3;

    assertEquals(expectedElements, actualElements);
  }

  @Test
  public void testCountElements_big() {
    int[] dimensions = new int[]{32, 45, 2, 34, 123, 3, 10};
    int actualElements = Tensor.countElements(dimensions);
    int expectedElements = 32 * 45 * 2 * 34 * 123 * 3 * 10;

    assertEquals(expectedElements, actualElements);
  }

  @Test
  public void testToString() {
    Vector testVector = new Vector(new double[]{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 0.10, 0.11, 0.12,
        0.13, 0.14, 0.15, 0.16,

        0.17, 0.18, 0.19, 0.20,
        0.21, 0.22, 0.23, 0.24,
        0.25, 0.26, 0.27, 0.28,
        0.29, 0.30, 0.31, 0.32,

        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 0.10, 0.11, 0.12,
        0.13, 0.14, 0.15, 0.16,

        0.17, 0.18, 0.19, 0.20,
        0.21, 0.22, 0.23, 0.24,
        0.25, 0.26, 0.27, 0.28,
        0.29, 0.30, 0.31, 0.32,
    });
    Tensor test = new Tensor(testVector, new int[]{4, 4, 2, 2});

    System.out.println(test);
  }
}
