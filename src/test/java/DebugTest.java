import com.jace.layer.ConvolutionLayer;
import com.jace.layer.LeakyRectifierLayer;
import com.jace.layer.MaxPooling2DLayer;
import com.jace.learner.NeuralNetwork;
import com.jace.math.Tensor;
import com.jace.math.Vector;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class DebugTest extends BaseTest {

  @Test
  public void testDebugSpew() {
    ConvolutionLayer layer = getFirstLayer();
    ConvolutionLayer layer0 = getConvolutionLayer();
    LeakyRectifierLayer layer1 = getLeakyRectifierLayer();
    MaxPooling2DLayer layer2 = getMaxPooling2DLayer();

    NeuralNetwork neuralNetwork = new NeuralNetwork();
    neuralNetwork.addLayer(layer);
    neuralNetwork.addLayer(layer0);
    neuralNetwork.addLayer(layer1);
    neuralNetwork.addLayer(layer2);
    neuralNetwork.setLearningRate(0.01);

    Vector inputVector = new Vector(new double[]{
        0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1, 1.1,
        1.2, 1.3, 1.4, 1.5
    });
    Tensor input = new Tensor(inputVector, new int[]{4, 4, 1});

    Vector targetVector = new Vector(new double[]{
        0.7, 0.6,
        0.5, 0.4,

        0.3, 0.2,
        0.1, 0
    });
    Tensor target = new Tensor(targetVector, new int[]{2, 2, 2});

    neuralNetwork.predict(input);
    neuralNetwork.backPropagate(target);
    neuralNetwork.updateGradient(input);
    neuralNetwork.updateWeights();

    checkFirstLayer(layer);
    checkLayer0(layer0);
    checkLayer1(layer1);
    checkLayer2(layer2);
  }

  private ConvolutionLayer getFirstLayer() {
    ConvolutionLayer layer = new ConvolutionLayer(new int[]{4, 4}, new int[]{3, 3, 1}, new int[]{4, 4, 1});
    layer.initialize();

    Vector filterVector = new Vector(new double[]{
        0.01, 0.02, 0.03,
        0.04, 0.05, 0.06,
        0.07, 0.08, 0.09,
    });
    Tensor filter = new Tensor(filterVector, new int[]{3, 3, 1});

    layer.setWeights(filter);

    Vector bias = new Vector(new double[]{0});
    layer.setBias(bias);

    return layer;
  }

  private ConvolutionLayer getConvolutionLayer() {
    ConvolutionLayer testLayer = new ConvolutionLayer(
        new int[]{4, 4},      // Input size (3rd dimension value is implicitly 1
        new int[]{3, 3, 2},   // Filter size
        new int[]{4, 4, 2});  // Output size
    testLayer.initialize();

    Vector filterVector = new Vector(new double[]{
        0.11, 0.12, 0.13,
        0.14, 0.15, 0.16,
        0.17, 0.18, 0.19,

        0.21, 0.22, 0.23,
        0.24, 0.25, 0.26,
        0.27, 0.28, 0.29
    });

    Tensor filter = new Tensor(filterVector, new int[]{3, 3, 2});
    testLayer.setWeights(filter);

    testLayer.setBias(new Vector(new double[]{0.1, 0.2}));

    return testLayer;
  }

  private LeakyRectifierLayer getLeakyRectifierLayer() {
    return new LeakyRectifierLayer(4 * 4 * 2);
  }

  private MaxPooling2DLayer getMaxPooling2DLayer() {
    return new MaxPooling2DLayer(new int[]{4, 4, 2});
  }

  public void checkFirstLayer(ConvolutionLayer layer) {
    Vector expectedActivationVector = new Vector(new double[]{
        0.083, 0.139, 0.178, 0.121,
        0.198, 0.303, 0.348, 0.225,
        0.33, 0.483, 0.528, 0.333,
        0.181, 0.253, 0.274, 0.163
    });
    Tensor expectedActivation = new Tensor(expectedActivationVector, new int[]{4, 4, 1});

    Vector expectedBlameVector = new Vector(new double[]{
        -0.1021612, -0.2424868, -0.2526264, -0.1485652,
        -0.2912204, -0.6655076, -0.6947076, -0.3948608,
        -0.3290468, -0.7531076, -0.7823076, -0.4446344,
        -0.2285932, -0.5069644, -0.5260248, -0.2907052
    });
    Tensor expectedBlame = new Tensor(expectedBlameVector, new int[]{4, 4, 1});

    Vector expectedWeightsVector = new Vector(new double[]{
        -0.0127731944, -0.0109769472, 0.0017403532,
        -0.0035582312, -0.005801104, 0.0112854192,
        0.0259183208, 0.0248525472, 0.0427923164,
    });
    Tensor expectedWeights = new Tensor(expectedWeightsVector, new int[]{3, 3, 1});

    Vector expectedBias = new Vector(new double[]{-0.0665352});

    assertVectorEquals(expectedActivation, layer.getActivation());
    assertVectorEquals(expectedBlame, layer.getBlame());
    assertVectorEquals(expectedWeights, layer.getWeights());
    assertVectorEquals(expectedBias, layer.getBias());
  }

  public void checkLayer0(ConvolutionLayer layer) {
    Vector expectedActivationVector0 = new Vector(new double[]{
        0.2279, 0.31527, 0.32242, 0.24273,
        0.35738, 0.52116, 0.52342, 0.36627,
        0.37058, 0.53488, 0.52774, 0.36507,
        0.27002, 0.37003, 0.36238, 0.26085,

        0.4002, 0.54017, 0.55382, 0.42993,
        0.61098, 0.88016, 0.88922, 0.63957,
        0.64538, 0.92468, 0.91874, 0.65217,
        0.49472, 0.67493, 0.66578, 0.49065
    });
    Tensor expectedActivation0 = new Tensor(expectedActivationVector0, new int[]{4, 4, 2});

    Vector expectedBlameVector0 = new Vector(new double[]{
        0, 0, 0, 0,
        0, 0.17884, 0.07658, 0,
        0, -0.03488, -0.12774, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, -0.58016, -0.68922, 0,
        0, -0.82468, -0.91874, 0,
        0, 0, 0, 0
    });
    Tensor expectedBlame0 = new Tensor(expectedBlameVector0, new int[]{4, 4, 2});

    Vector expectedWeightsVector0 = new Vector(new double[]{
        0.1097987688, 0.1198346784, 0.1300021996,
        0.1398540524, 0.149965446, 0.1601851276,
        0.1705737384, 0.1808298856, 0.1908954992,

        0.2041438028, 0.2122707704, 0.2231963076,
        0.2296039884, 0.237009478, 0.2490165836,
        0.2609394564, 0.2689549576, 0.2808844832
    });
    Tensor expectedWeights0 = new Tensor(expectedWeightsVector0, new int[]{3, 3, 2});

    Vector expectedBias0 = new Vector(new double[]{0.100928, 0.169872});

    assertVectorEquals(expectedActivation0, layer.getActivation());
    assertVectorEquals(expectedBlame0, layer.getBlame());
    assertVectorEquals(expectedWeights0, layer.getWeights());
    assertVectorEquals(expectedBias0, layer.getBias());
  }

  private void checkLayer1(LeakyRectifierLayer layer) {
    Vector expectedBlameVector1 = new Vector(new double[]{
        0, 0, 0, 0,
        0, 0.17884, 0.07658, 0,
        0, -0.03488, -0.12774, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, -0.58016, -0.68922, 0,
        0, -0.82468, -0.91874, 0,
        0, 0, 0, 0
    });
    Tensor expectedBlame1 = new Tensor(expectedBlameVector1, new int[]{4, 4, 2});

    Vector expectedActivationVector1 = new Vector(new double[]{
        0.2279, 0.31527, 0.32242, 0.24273,
        0.35738, 0.52116, 0.52342, 0.36627,
        0.37058, 0.53488, 0.52774, 0.36507,
        0.27002, 0.37003, 0.36238, 0.26085,

        0.4002, 0.54017, 0.55382, 0.42993,
        0.61098, 0.88016, 0.88922, 0.63957,
        0.64538, 0.92468, 0.91874, 0.65217,
        0.49472, 0.67493, 0.66578, 0.49065
    });
    Tensor expectedActivation1 = new Tensor(expectedActivationVector1, new int[]{4, 4, 2});

    assertVectorEquals(expectedActivation1, layer.getActivation());
    assertVectorEquals(expectedBlame1, layer.getBlame());
  }

  private void checkLayer2(MaxPooling2DLayer layer) {
    Vector expectedBlameVector2 = new Vector(new double[]{
        0.17884, 0.07658,
        -0.03488, -0.12774,

        -0.58016, -0.68922,
        -0.82468, -0.91874
    });
    Tensor expectedBlame2 = new Tensor(expectedBlameVector2, new int[]{2, 2, 2});

    Vector expectedActivationVector2 = new Vector(new double[]{
        0.52116, 0.52342,
        0.53488, 0.52774,

        0.88016, 0.88922,
        0.92468, 0.91874
    });
    Tensor expectedActivation2 = new Tensor(expectedActivationVector2, new int[]{2, 2, 2});

    assertVectorEquals(expectedActivation2, layer.getActivation());
    assertVectorEquals(expectedBlame2, layer.getBlame());
  }
}
