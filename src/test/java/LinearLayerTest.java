import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(JUnit4.class)
public class LinearLayerTest extends BaseTest {

  @Test
  public void activate() {
    Vector x = new Vector(new double[]{0, 1, 2});

    Vector weightsVector = new Vector(new double[]{1, 2, 3, 2, 1, 0});
    Matrix weights = Matrix.deserialize(weightsVector, 2, 3);

    Vector bias = new Vector(new double[]{1, 5});

    LinearLayer linearLayer = new LinearLayer(3, 2);
    linearLayer.setWeights(weights);
    linearLayer.setBias(bias);

    linearLayer.activate(x);

    Vector expectedResult = new Vector(new double[]{9, 6});

    assertEquals(expectedResult,  linearLayer.getActivation());
  }

  @Test
  public void backPropagate() {
    LinearLayer linearLayer = new LinearLayer(3, 2);
    Vector weightsVector = new Vector(new double[] {1, 2, 3, 2, 1, 0});
    Matrix weights = Matrix.deserialize(weightsVector, 2, 3);
    Vector bias = new Vector(new double[] {1, 5});
    Vector blame = new Vector(new double[] {0.75, 2});

    linearLayer.setWeights(weights);
    linearLayer.setBias(bias);
    linearLayer.setBlame(blame);

    Vector expectedPreviousBlame = new Vector(new double[] {4.75, 3.5, 2.25});
    Vector actualPreviousBlame = linearLayer.backPropagate();

    assertEquals(expectedPreviousBlame, actualPreviousBlame);
  }

  @Test
  public void updateGradient() {

  }

  @Test
  public void ordinaryLeastSquares() {
    int data_rows = 100;
    int x_features = 13;
    int y_features = 4;

    double standardDeviation = 1;

    Matrix X = getRandomMatrix(data_rows, x_features);
    Matrix Y = new Matrix(data_rows, y_features);

    Matrix randomWeights = getRandomMatrix(y_features, x_features);
    Vector randomBias = getRandomVector(y_features);

    LinearLayer linearLayer = new LinearLayer(x_features, y_features);
    linearLayer.setWeights(randomWeights);
    linearLayer.setBias(randomBias);

    for (int i = 0; i < data_rows; i++) {
      linearLayer.activate(X.row(i));
      Vector output_y = linearLayer.getActivation();
      Y.setRow(i, output_y);
    }

    addRandomNoiseToMatrix(Y, standardDeviation);

    linearLayer.ordinaryLeastSquares(X, Y);

    Matrix newWeights = linearLayer.getWeights();
    Vector newBias = linearLayer.getBias();

    double error = randomWeights.errorAgainst(newWeights);
    error += randomBias.errorAgainst(newBias);

    double averageError = error / (double) (x_features * y_features + y_features);

    assertTrue(averageError < standardDeviation);
  }
}
