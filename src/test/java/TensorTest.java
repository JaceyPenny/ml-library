import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.assertEquals;

@RunWith(JUnit4.class)
public class TensorTest extends BaseTest {

  @Test
  public void testCountElements_small() throws Exception {
    int[] dimensions = new int[]{1, 2, 3};
    int actualElements = Tensor.countElements(dimensions);
    int expectedElements = 1 * 2 * 3;

    assertEquals(expectedElements, actualElements);
  }

  @Test
  public void testCountElements_big() throws Exception {
    int[] dimensions = new int[]{32, 45, 2, 34, 123, 3, 10};
    int actualElements = Tensor.countElements(dimensions);
    int expectedElements = 32 * 45 * 2 * 34 * 123 * 3 * 10;

    assertEquals(expectedElements, actualElements);
  }
}
