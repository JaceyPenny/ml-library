import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.*;

@RunWith(JUnit4.class)
public class LayerLinearTest extends BaseTest {

    @Test
    public void activate() {
        Vector x = new Vector(new double[] {0, 1, 2});
        Vector weights = new Vector(new double[] {1, 5, 1, 2, 3, 2, 1, 0});

        LayerLinear layerLinear = new LayerLinear(3, 2);
        layerLinear.activate(weights, x);

        Vector expectedResult = new Vector(new double[] {9, 6});

        assertTrue(layerLinear.activation.equals(expectedResult));
    }

    @Test
    public void ordinaryLeastSquares() {
        int data_rows = 100;
        int x_features = 13;
        int y_features = 4;

        double standardDeviation = 1;

        Matrix X = getRandomMatrix(data_rows, x_features);
        Matrix Y = new Matrix(data_rows, y_features);

        Vector randomWeights = getRandomVector(y_features + x_features * y_features);

        LayerLinear layerLinear = new LayerLinear(x_features, y_features);

        for (int i = 0; i < data_rows; i++) {
            layerLinear.activate(randomWeights, X.row(i));
            Vector output_y = layerLinear.activation;
            Y.setRow(i, output_y);
        }

        addRandomNoiseToMatrix(Y, standardDeviation);
        Vector newWeights = new Vector(randomWeights.len);
        layerLinear.ordinaryLeastSquares(X, Y, newWeights);

        double error = 0;
        for (int i = 0; i < randomWeights.len; i++) {
            error += Math.abs(randomWeights.get(i) - newWeights.get(i));
        }

        double averageError = error / (double) (x_features * y_features);

        assertTrue(averageError < standardDeviation);
    }
}
