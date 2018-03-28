import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class MaxPooling2DLayerTest extends BaseTest {

  @Test
  public void testDebugSpew() {
    MaxPooling2DLayer testLayer = new MaxPooling2DLayer(new int[]{4, 4, 2});

    Vector inputVector = new Vector(new double[]{
        0.083, 0.139, 0.178, 0.121,
        0.198, 0.303, 0.348, 0.225,
        0.33, 0.483, 0.528, 0.333,
        0.181, 0.253, 0.274, 0.163,

        0.283, 0.419, 0.518, 0.401,
        0.568, 0.853, 0.988, 0.715,
        0.94, 1.393, 1.528, 1.063,
        0.701, 1.013, 1.094, 0.763
    });
    Tensor input = new Tensor(inputVector, new int[]{4, 4, 2});

    testLayer.activate(input);

    Vector expectedActivationVector = new Vector(new double[]{
        0.303, 0.348,
        0.483, 0.528,

        0.853, 0.988,
        1.393, 1.528
    });
    Tensor expectedActivation = new Tensor(expectedActivationVector, new int[]{2, 2, 2});

    assertVectorEquals(expectedActivation, testLayer.getActivation(), 1e-4);

    Vector thisBlameVector = new Vector(new double[]{
        0.397,0.252,
        0.017,-0.128,

        -0.553,-0.788,
        -1.293,-1.528
    });
    Tensor thisBlame = new Tensor(thisBlameVector, new int[]{2, 2, 2});
    testLayer.setBlame(thisBlame);

    Vector expectedBlameVector = new Vector(new double[]{
        0, 0, 0, 0,
        0, 0.397, 0.252, 0,
        0, 0.017, -0.128, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, -0.553, -0.788, 0,
        0, -1.293, -1.528, 0,
        0, 0, 0, 0
    });
    Tensor expectedBlame = new Tensor(expectedBlameVector, new int[]{4, 4, 2});

    Vector actualBlame = testLayer.backPropagate();
    assertVectorEquals(expectedBlame, actualBlame, 1e-4);
  }
}
