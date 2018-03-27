import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.assertEquals;

@RunWith(JUnit4.class)
public class ConvolutionLayerTest extends BaseTest {

  @Test
  public void testDebugSpew() {
    ConvolutionLayer testLayer = new ConvolutionLayer(
        new int[]{4, 4},      // Input size (3rd dimension value is implicitly 1
        new int[]{3, 3, 2},   // Filter size
        new int[]{4, 4, 2});  // Output size

    Vector filterVector = new Vector(new double[]{
        0.01, 0.02, 0.03,
        0.04, 0.05, 0.06,
        0.07, 0.08, 0.09,

        0.11, 0.12, 0.13,
        0.14, 0.15, 0.16,
        0.17, 0.18, 0.19
    });

    Tensor filter = new Tensor(filterVector, new int[]{3, 3, 2});
    testLayer.setFilter(filter);

    Vector inputVector = new Vector(new double[]{
        0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1, 1.1,
        1.2, 1.3, 1.4, 1.5
    });
    Tensor input = new Tensor(inputVector, new int[]{4, 4, 1});

    testLayer.activate(input);

    Vector expectedActivationVector = new Vector(new double[]{
        0.283, 0.419, 0.518, 0.401,
        0.568, 0.853, 0.988, 0.715,
        0.94, 1.393, 1.528, 1.063,
        0.701, 1.013, 1.094, 0.763,

        0.083, 0.139, 0.178, 0.121,
        0.198, 0.303, 0.348, 0.225,
        0.33, 0.483, 0.528, 0.333,
        0.181, 0.253, 0.274, 0.163
    });
    Tensor expectedActivation = new Tensor(expectedActivationVector, new int[]{4, 4, 2});

    assertEquals(expectedActivation, testLayer.getActivation());
  }

  private static void assertActivationEquals(Vector first, Vector second) throws AssertionError {

  }
}
