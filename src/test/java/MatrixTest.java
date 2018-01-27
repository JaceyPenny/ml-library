import org.junit.Test;

import static org.junit.Assert.*;

public class MatrixTest extends BaseTest {
    @Test
    public void deserialize() {
        Matrix expectedMatrix = getSampleMatrix(4, 6);
        Vector serializedMatrix = expectedMatrix.serialize();

        Matrix deserializedMatrix = Matrix.deserialize(serializedMatrix, expectedMatrix.rows(), expectedMatrix.cols());

        assertEquals(deserializedMatrix, expectedMatrix);
    }

    @Test
    public void serialize() {
        Matrix sampleMatrix = getSampleMatrix(4, 6);

        Vector output = sampleMatrix.serialize();

        Vector expectedResult = getSampleVector(4 * 6);

        assertEquals(expectedResult, output);
    }

    @Test
    public void column() {
        Matrix sampleMatrix = getSampleMatrix(3, 3);

        Vector column0 = sampleMatrix.column(0);
        Vector column1 = sampleMatrix.column(1);
        Vector column2 = sampleMatrix.column(2);

        Vector expected0 = new Vector(new double[] {0, 3, 6});
        Vector expected1 = new Vector(new double[] {1, 4, 7});
        Vector expected2 = new Vector(new double[] {2, 5, 8});

        assertEquals(expected0, column0);
        assertEquals(expected1, column1);
        assertEquals(expected2, column2);
    }
}