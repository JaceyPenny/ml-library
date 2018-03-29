import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
    testLayer.setWeights(filter);

    testLayer.setBias(new Vector(new double[]{0.0, 0.1}));

    Vector inputVector = new Vector(new double[]{
        0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1, 1.1,
        1.2, 1.3, 1.4, 1.5
    });
    Tensor input = new Tensor(inputVector, new int[]{4, 4, 1});

    testLayer.activate(input);

    Vector expectedActivationVector = new Vector(new double[]{
        0.083, 0.139, 0.178, 0.121,
        0.198, 0.303, 0.348, 0.225,
        0.33, 0.483, 0.528, 0.333,
        0.181, 0.253, 0.274, 0.163,

        0.283, 0.419, 0.518, 0.401,
        0.568, 0.853, 0.988, 0.715,
        0.94, 1.393, 1.528, 1.063,
        0.701, 1.013, 1.094, 0.763,
    });
    Tensor expectedActivation = new Tensor(expectedActivationVector, new int[]{4, 4, 2});
    assertVectorEquals(expectedActivation, testLayer.getActivation(), 1e-4);

    Vector blameVector = new Vector(new double[]{
        0, 0, 0, 0,
        0, 0.397, 0.252, 0,
        0, 0.017, -0.128, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, -0.553, -0.788, 0,
        0, -1.293, -1.528, 0,
        0, 0, 0, 0
    });
    Tensor blame = new Tensor(blameVector, new int[]{4, 4, 2});

    testLayer.setBlame(blame);
    testLayer.updateGradient(input);

    Vector expectedGradientVector = new Vector(new double[]{
        -0.032, 0.0218, 0.0756,
        0.1832, 0.237, 0.2908,
        0.3984, 0.4522, 0.506,

        -1.36, -1.7762, -2.1924,
        -3.0248, -3.441, -3.8572,
        -4.6896, -5.1058, -5.522
    });
    Tensor expectedGradient = new Tensor(expectedGradientVector, new int[]{3, 3, 2});

    Vector expectedBiasGradient = new Vector(new double[]{0.538, -4.162});

    assertVectorEquals(expectedGradient, testLayer.getGradient(), 1e-4);
    assertVectorEquals(expectedBiasGradient, testLayer.getBiasGradient(), 1e-4);

    Vector expectedWeightsVector = new Vector(new double[]{
        0.00968, 0.020218, 0.030756,
        0.041832, 0.05237, 0.062908,
        0.073984, 0.084522, 0.09506,

        0.0964, 0.102238, 0.108076,
        0.109752, 0.11559, 0.121428,
        0.123104, 0.128942, 0.13478
    });
    Tensor expectedWeights = new Tensor(expectedWeightsVector, new int[]{3, 3, 2});

    testLayer.applyGradient(0.01, 0);
    Tensor actualWeights = testLayer.getFilter();

    assertVectorEquals(expectedWeights, actualWeights, 1e-5);
  }
}
