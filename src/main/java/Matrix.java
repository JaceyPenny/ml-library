import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * This stores a matrix, A.K.A. data set, A.K.A. table. Each element is
 * represented as a double value. Nominal values are represented using their
 * corresponding zero-indexed enumeration value. For convenience,
 * the matrix also stores some meta-data which describes the columns (or attributes)
 * in the matrix.
 */
public class Matrix {
  public enum VectorType {
    ROW, COLUMN
  }

  /**
   * Used to represent elements in the matrix for which the value is not known.
   */
  public static final double UNKNOWN_VALUE = -1e308;

  private ArrayList<double[]> data = new ArrayList<>(); //matrix elements
  private Metadata metadata;

  /**
   * Creates a 0x0 matrix. (Next, to give this matrix some dimensions, you should call:
   * loadARFF
   * setSize
   * newColumn, or
   * copyMetaData
   */
  public Matrix() {
    metadata = new Metadata();
  }

  public Matrix(int rows, int cols) {
    this();
    setSize(rows, cols);
  }

  public Matrix(Matrix that) {
    metadata = new Metadata(that.getMetadata().getFileName());
    setSize(that.rows(), that.cols());
    copyBlock(0, 0, that, 0, 0, that.rows(), that.cols()); // (copies the meta data too)
  }

  public Matrix(Vector vector, VectorType vectorType) {
    this(vectorType == VectorType.ROW ? 1 : vector.size(), vectorType == VectorType.ROW ? vector.size() : 1);

    if (vectorType == VectorType.ROW) {
      setRow(0, vector);
    } else {
      setColumn(0, vector);
    }
  }

  public Matrix(Json json) {
    int rowCount = json.size();
    int colCount = json.get(0).size();
    metadata = new Metadata();
    setSize(rowCount, colCount);
    for (int i = 0; i < rowCount; i++) {
      Json jsonRow = json.get(i);
      double[] row = data.get(i);
      for (int j = 0; j < colCount; j++) {
        row[j] = jsonRow.getDouble(j);
      }
    }
  }

  /**
   * Marshals this object into a Json DOM
   */
  public Json marshal() {
    Json jsonList = Json.newList();
    for (int i = 0; i < rows(); i++) {
      double[] row = data.get(i);
      Json jsonRow = Json.newList();

      for (double value : row) {
        jsonRow.add(value);
      }

      jsonList.add(jsonRow);
    }

    return jsonList;
  }

  public static Matrix fromARFF(String fileName) {
    Matrix matrix = new Matrix();
    matrix.loadARFF(fileName);
    return matrix;
  }

  /**
   * Loads the matrix from an ARFF file
   */
  public void loadARFF(String fileName) {
    int attributeCount = 0;
    Scanner scanner = null;
    metadata.clear();

    try {
      scanner = new Scanner(new File(fileName));
      while (scanner.hasNextLine()) {
        String line = scanner.nextLine().trim();
        String upper = line.toUpperCase();

        if (upper.startsWith("@RELATION")) {
          String parsedFileName = line.split(" ")[1];
          metadata.setFileName(parsedFileName);
        } else if (upper.startsWith("@ATTRIBUTE")) {
          metadata.parseAttribute(line);
          attributeCount++;
        } else if (upper.startsWith("@DATA")) {
          data.clear();

          while (scanner.hasNextLine()) {
            line = scanner.nextLine().trim();
            if (line.startsWith("%") || line.isEmpty()) {
              continue;
            }

            double[] row = new double[attributeCount];
            data.add(row);

            Json.StringParser stringParser = new Json.StringParser(line);
            for (int i = 0; i < attributeCount; i++) {
              stringParser.skipWhitespace();
              String parsedValue = stringParser.untilQuoteSensitive(',', '\n');

              boolean attributeIsContinuous = metadata.attributeIsContinuous(i);
              if (parsedValue.equals("?")) { // Unknown values are always set to UNKNOWN_VALUE
                row[i] = UNKNOWN_VALUE;
              } else if (!attributeIsContinuous) { // if it's nominal
                row[i] = metadata.getValueForAttributeInColumn(parsedValue, i);
              } else { // else it's continuous
                row[i] = Double.parseDouble(parsedValue); // The attribute is continuous
              }

              stringParser.advance(1);
            }
          }
        }
      }
    } catch (FileNotFoundException e) {
      throw new IllegalArgumentException("Failed to open file: " + fileName + ".");
    } finally {
      if (scanner != null) {
        scanner.close();
      }
    }
  }

  public Metadata getMetadata() {
    return metadata;
  }

  public void printRow(double[] row, PrintWriter outputStream) {
    if (row.length != cols()) {
      throw new RuntimeException("Unexpected row size");
    }

    for (int j = 0; j < row.length; j++) {
      if (row[j] == UNKNOWN_VALUE) {
        outputStream.print("?");
      } else {
        int valueCount = metadata.getEnumSizeForAttributeInColumn(j);
        if (valueCount == 0) {
          if (Math.floor(row[j]) == row[j]) {
            outputStream.print((int) Math.floor(row[j]));
          } else {
            outputStream.print(row[j]);
          }
        } else {
          int value = (int) row[j];
          if (value >= valueCount) {
            throw new IllegalArgumentException("Value out of range.");
          }

          outputStream.print(metadata.getAttributeForValueInColumn(value, j));
        }
      }

      if (j + 1 < cols()) {
        outputStream.print(",");
      }
    }
  }

  /**
   * Saves the matrix to an ARFF file
   */
  public void saveARFF(String filename) {
    PrintWriter outputStream = null;

    try {
      outputStream = new PrintWriter(filename);
      // Print the relation name, if one has been provided ('x' is default)
      outputStream.print("@RELATION ");
      outputStream.println(metadata.getFileName().isEmpty() ? "x" : metadata.getFileName());

      // Print each attribute in order
      for (int i = 0; i < metadata.getAttributeNames().size(); i++) {
        outputStream.print("@ATTRIBUTE ");

        String attributeName = metadata.getAttributeNames().get(i);
        outputStream.print(attributeName.isEmpty() ? "x" : attributeName);

        if (metadata.attributeIsContinuous(i)) {
          outputStream.println(" REAL");
        } else {
          int values = metadata.getEnumSizeForAttributeInColumn(i);

          outputStream.print(" {");
          for (int j = 0; j < values; j++) {
            outputStream.print(metadata.getAttributeForValueInColumn(j, i));
            if (j + 1 < values) {
              outputStream.print(",");
            }
          }
          outputStream.println("}");
        }
      }

      // Print the data
      outputStream.println("@DATA");
      for (int i = 0; i < rows(); i++) {
        double[] row = data.get(i);
        printRow(row, outputStream);
        outputStream.println();
      }
    } catch (FileNotFoundException e) {
      throw new IllegalArgumentException("Error creating file: " + filename + ".");
    } finally {
      if (outputStream != null) {
        outputStream.close();
      }
    }
  }

  /**
   * Makes a rows-by-columns matrix of *ALL CONTINUOUS VALUES*.
   * This method wipes out any data currently in the matrix. It also
   * wipes out any meta-data.
   */
  public void setSize(int rows, int cols) {
    data.clear();

    // Set the meta-data
    metadata.clear();

    // Make space for each of the columns, then each of the rows
    newColumns(cols);
    newRows(rows);
  }

  /**
   * Clears this matrix and copies the meta-data from that matrix.
   * In other words, it makes a zero-row matrix with the same number
   * of columns as "that" matrix. You will need to call newRow or newRows
   * to give the matrix some rows.
   */
  public void copyMetaData(Matrix that) {
    data.clear();
    metadata = that.getMetadata().copy();
  }

  /**
   * Adds a column with the specified name
   */
  public void newColumn(String name) {
    data.clear();
    metadata.addColumn(name);
  }

  /**
   * Adds a column to this matrix with the specified number of values. (Use 0 for
   * a continuous attribute.) This method also sets the number of rows to 0, so
   * you will need to call newRow or newRows when you are done adding columns.
   */
  public void newColumn(int valueCount) {
    data.clear();
    metadata.addColumn(valueCount);
  }

  /**
   * Adds a column to this matrix with 0 values (continuous data).
   */
  public void newColumn() {
    this.newColumn(0);
  }

  /**
   * Adds n columns to this matrix, each with 0 values (continuous data).
   */
  public void newColumns(int n) {
    for (int i = 0; i < n; i++) {
      newColumn();
    }
  }

  /**
   * Returns the index of the specified value in the specified column.
   * If there is no such value, adds it to the column.
   */
  public int findOrCreateValue(int column, String attribute) {
    return metadata.findOrCreateValueForAttributeInColumn(attribute, column);
  }

  /**
   * Adds one new row to this matrix. Returns a reference to the new row.
   */
  public double[] newRow() {
    int columns = cols();
    if (columns == 0) {
      throw new IllegalArgumentException("You must add some columns before you add any rows.");
    }

    double[] newRow = new double[columns];
    data.add(newRow);
    return newRow;
  }

  /**
   * Adds one new row to this matrix at the specified location. Returns a reference to the new row.
   */
  public double[] insertRow(int i) {
    int columns = cols();
    if (columns == 0) {
      throw new IllegalArgumentException("You must add some columns before you add any rows.");
    }

    double[] newRow = new double[columns];
    data.add(i, newRow);
    return newRow;
  }

  /**
   * Removes the specified row from this matrix. Returns a reference to the removed row.
   */
  public double[] removeRow(int i) {
    return data.remove(i);
  }

  /**
   * Appends the specified row to this matrix.
   */
  public void takeRow(double[] row) {
    if (row.length != cols()) {
      throw new IllegalArgumentException("Row size differs from the number of columns in this matrix.");
    }

    data.add(row);
  }

  public void setColumn(int index, Vector value) {
    setColumn(index, value.values);
  }

  public void setColumn(int index, double[] values) {
    if (index > this.cols()) {
      throw new IllegalArgumentException("Invalid index " + index + " in matrix.");
    } else if (values.length != this.rows()) {
      throw new IllegalArgumentException("Invalid column length (" + values.length + ") for matrix.");
    }

    for (int i = 0; i < rows(); i++) {
      row(i).set(index, values[i]);
    }
  }

  public void setRow(int index, Vector row) {
    setRow(index, row.toDoubleArray());
  }

  public void setRow(int index, double[] row) {
    if (index >= this.rows()) {
      throw new IllegalArgumentException("Invalid index " + index + " in matrix.");
    } else if (row.length != this.cols()) {
      throw new IllegalArgumentException("Invalid row length (" + row.length + ") for matrix.");
    }

    System.arraycopy(row, 0, data.get(index), 0, row.length);
  }

  /**
   * Adds "n" new rows to the Matrix
   */
  public void newRows(int n) {
    for (int i = 0; i < n; i++) {
      newRow();
    }
  }

  /**
   *  Returns the number of rows in the matrix
   */
  public int rows() {
    return data.size();
  }

  /**
   * Returns the number of columns (or attributes) in the matrix
   */
  public int cols() {
    return metadata.getAttributeNames().size();
  }

  public String getString(int row, int column) {
    double value = data.get(row)[column];
    return metadata.getAttributeForValueInColumn((int) value, column);
  }

  private void checkIndex(int row, int column) {
    if (row < 0 || row >= rows()) {
      throw new IllegalArgumentException("Invalid row index: " + row);
    }

    if (column < 0 || column >= cols()) {
      throw new IllegalArgumentException("Invalid column index: " + column);
    }
  }

  public double get(int row, int column) {
    checkIndex(row, column);

    return row(row).get(column);
  }

  public void set(int row, int column, double value) {
    checkIndex(row, column);

    row(row).set(column, value);
  }

  /**
   * Returns a reference to the specified row
   */
  public Vector row(int index) {
    return new Vector(data.get(index));
  }

  public Vector column(int index) {
    Vector column = new Vector(rows());
    for (int i = 0; i < rows(); i++) {
      column.set(i, row(i).get(index));
    }
    return column;
  }

  public void shuffleRows() {
    Random random = new Random();
    for (int i = rows(); i >= 2; i--) {
      int r = random.nextInt(i);
      swapRows(i - 1, r);
    }
  }

  /**
   * Swaps the positions of the two specified rows
   */
  public void swapRows(int a, int b) {
    double[] temp = data.get(a);
    data.set(a, data.get(b));
    data.set(b, temp);
  }

  /**
   * Copies that matrix
   */
  void copy(Matrix that) {
    setSize(that.rows(), that.cols());
    copyBlock(0, 0, that, 0, 0, that.rows(), that.cols());
  }

  /**
   * Returns the mean of the elements in the specified column.
   * (Elements with the value UNKNOWN_VALUE are ignored)
   */
  public double columnMean(int column) {
    double sum = 0.0;
    int count = 0;
    for (double[] list : data) {
      double val = list[column];
      if (val != UNKNOWN_VALUE) {
        sum += val;
        count++;
      }
    }

    return sum / count;
  }


  /**
   * Returns the minimum element in the specified column.
   * (Elements with the value UNKNOWN_VALUE are ignored.)
   */
  public double columnMin(int col) {
    double min = Double.MAX_VALUE;
    for (double[] list : data) {
      double val = list[col];
      if (val != UNKNOWN_VALUE) {
        min = Math.min(min, val);
      }
    }

    return min;
  }


  /**
   * Returns the maximum element in the specified column.
   * (Elements with the value UNKNOWN_VALUE are ignored.)
   */
  public double columnMax(int col) {
    double max = -Double.MAX_VALUE;
    for (double[] list : data) {
      double val = list[col];
      if (val != UNKNOWN_VALUE) {
        max = Math.max(max, val);
      }
    }

    return max;
  }

  /**
   * Returns the most common value in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
   */
  public double mostCommonValue(int col) {
    HashMap<Double, Integer> counts = new HashMap<>();
    for (double[] list : data) {
      double val = list[col];
      if (val != UNKNOWN_VALUE) {
        int result = counts.getOrDefault(val, 0);
        counts.put(val, result + 1);
      }
    }

    int valueCount = 0;
    double value = 0;
    for (Map.Entry<Double, Integer> entry : counts.entrySet()) {
      if (entry.getValue() > valueCount) {
        value = entry.getKey();
        valueCount = entry.getValue();
      }
    }

    return value;
  }

  /**
   * Copies the specified rectangular portion of that matrix, and puts it in the specified location in this matrix.
   */
  public void copyBlock(
      int destRow,
      int destCol,
      Matrix that,
      int rowBegin,
      int colBegin,
      int rowCount,
      int colCount) {
    if (destRow + rowCount > this.rows() || destCol + colCount > this.cols()) {
      throw new IllegalArgumentException("Out of range for destination matrix.");
    } else if (rowBegin + rowCount > that.rows() || colBegin + colCount > that.cols()) {
      throw new IllegalArgumentException("Out of range for source matrix.");
    }

    // Copy the specified region of meta-data
    for (int i = 0; i < colCount; i++) {
      that.getMetadata().copyAttributeToMetadata(this.metadata, colBegin + i, destCol + i);
    }

    // Copy the specified region of data
    for (int i = 0; i < rowCount; i++) {
      double[] source = that.data.get(rowBegin + i);
      double[] dest = this.data.get(destRow + i);
      System.arraycopy(source, colBegin, dest, destCol, colCount);
    }
  }

  public Matrix copyBlock(int startRow, int startCol, int rows, int cols) {
    Matrix result = new Matrix(rows, cols);
    result.copyBlock(0, 0, this, startRow, startCol, rows, cols);
    return result;
  }

  /**
   * Returns a new matrix with just the rows between [rowStart, rowEnd)
   */
  public Matrix duplicateRows(int rowStart, int rowEnd) {
    Matrix result = new Matrix(rowEnd - rowStart, this.cols());
    result.copyBlock(0, 0, this, rowStart, 0, rowEnd - rowStart, this.cols());
    return result;
  }

  public Matrix toOneHot() {
    if (cols() != 1) {
      throw new IllegalStateException("Cannot convert a multi-column matrix to one-hot representation");
    }

    int maxValue = 0;

    for (int i = 0; i < rows(); i++) {
      Vector row = row(i);
      int value = (int) Math.round(row.get(0));
      if (value > maxValue) {
        maxValue = value;
      }
    }

    Matrix result = new Matrix(rows(), maxValue + 1);

    for (int i = 0; i < rows(); i++) {
      Vector row = row(i);
      int value = (int) Math.round(row.get(0));
      result.set(i, value, 1);
    }

    return result;
  }

  /**
   * Sets every element in the matrix to the specified value.
   */
  public void fill(double val) {
    for (double[] vec : data) {
      for (int i = 0; i < vec.length; i++)
        vec[i] = val;
    }
  }

  public void fill(Supplier<Double> supplier) {
    for (int i = 0; i < rows(); i++) {
      for (int j = 0; j < cols(); j++) {
        set(i, j, supplier.get());
      }
    }
  }

  /**
   * Scales every element in the matrix by the specified value
   */
  public void scale(double scalar) {
    for (double[] vec : data) {
      for (int i = 0; i < vec.length; i++)
        vec[i] *= scalar;
    }
  }

  public void add(Matrix that) {
    addScaled(that, 1);
  }

  /**
   * Adds every element in that matrix to this one
   */
  public void addScaled(Matrix that, double scalar) {
    if (that.rows() != this.rows() || that.cols() != this.cols())
      throw new IllegalArgumentException("Mismatching size");
    for (int i = 0; i < rows(); i++) {
      Vector dest = this.row(i);
      Vector src = that.row(i);
      dest.addScaled(src, scalar);
    }
  }

  /**
   * Sets this to the identity matrix.
   */
  public void setToIdentity() {
    fill(0.0);
    int m = Math.min(cols(), rows());
    for (int i = 0; i < m; i++)
      data.get(i)[i] = 1.0;
  }

  /**
   * Throws an exception if that has a different number of columns than
   * this, or if one of its columns has a different number of values.
   */
  public void checkCompatibility(Matrix that) {
    int columns = cols();
    if (that.cols() != columns) {
      throw new IllegalArgumentException("Matrices have different number of columns.");
    }

    for (int i = 0; i < columns; i++) {
      if (metadata.getEnumSizeForAttributeInColumn(i)
          != that.getMetadata().getEnumSizeForAttributeInColumn(i)) {
        throw new IllegalArgumentException("Column " + i + " has mis-matching number of values.");
      }
    }
  }

  /**
   * Deserializes a matrix from a vector given the matrix dimensions.
   * Matrices are deserialized row-wise, meaning that rows are assembled by taking contiguous chunks
   * from the input vector.
   * <br>
   * This operation <b>copies</b> the values from the input vector.
   */
  public static Matrix deserialize(Vector vector, int rows, int columns) {
    if (vector.size() < rows * columns) {
      throw new IllegalArgumentException("The supplied vector is too small to fill a matrix of size (" + rows + ", " + columns);
    }

    Matrix output = new Matrix(rows, columns);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        double value = vector.get(r * columns + c);
        output.set(r, c, value);
      }
    }

    return output;
  }

  public Vector serialize() {
    if (rows() == 0 || cols() == 0) {
      throw new IllegalStateException("Cannot serialize a Matrix with 0 rows or 0 columns.");
    }

    Vector outputVector = new Vector(rows() * cols());
    for (int i = 0; i < rows(); i++) {
      Vector row = row(i);
      outputVector.set(i * cols(), row);
    }

    return outputVector;
  }

  public double errorAgainst(Matrix other) {
    if (rows() != other.rows() || cols() != other.cols()) {
      throw new IllegalArgumentException("These matrices do not have the same size.");
    }

    double error = 0;
    for (int i = 0; i < rows(); i++) {
      for (int j = 0; j < cols(); j++) {
        error += Math.abs(get(i, j) - other.get(i, j));
      }
    }

    return error;
  }

  private static class SortComparator implements Comparator<double[]> {
    int column;
    boolean ascending;

    SortComparator(int col, boolean ascend) {
      column = col;
      ascending = ascend;
    }

    public int compare(double[] a, double[] b) {
      if (ascending) {
        if (a[column] < b[column])
          return -1;
        else if (a[column] > b[column])
          return 1;
        else
          return 0;
      } else {
        if (a[column] < b[column])
          return 1;
        else if (a[column] > b[column])
          return -1;
        else
          return 0;
      }
    }
  }

  public void sort(int column, boolean ascending) {
    data.sort(new SortComparator(column, ascending));
  }

  double pythag(double a, double b) {
    double at = Math.abs(a);
    double bt = Math.abs(b);
    if (at > bt) {
      double ct = bt / at;
      return at * Math.sqrt(1.0 + ct * ct);
    } else if (bt > 0.0) {
      double ct = at / bt;
      return bt * Math.sqrt(1.0 + ct * ct);
    } else
      return 0.0;
  }

  double safeDivide(double n, double d) {
    if (d == 0.0 && n == 0.0)
      return 0.0;
    else {
      double t = n / d;
      //GAssert(t > -1e200, "prob");
      return t;
    }
  }

  double takeSign(double a, double b) {
    return (b >= 0.0 ? Math.abs(a) : -Math.abs(a));
  }

  void fixNans() {
    int colCount = cols();
    for (int i = 0; i < rows(); i++) {
      double[] pRow = data.get(i);
      for (int j = 0; j < colCount; j++) {
        if (Double.isNaN(pRow[j]))
          pRow[j] = (i == j ? 1.0 : 0.0);
      }
    }
  }

  Matrix transpose() {
    Matrix res = new Matrix(cols(), rows());
    for (int i = 0; i < rows(); i++) {
      for (int j = 0; j < cols(); j++)
        res.data.get(j)[i] = data.get(i)[j];
    }
    return res;
  }

  /**
   * Swaps the the two specified columns
   */
  public void swapColumns(int a, int b) {
    for (int i = 0; i < rows(); i++) {
      double[] row = data.get(i);
      double valueTemp = row[a];
      row[a] = row[b];
      row[b] = valueTemp;
    }

    metadata.swapColumns(a, b);
  }

  static Matrix multiply(Matrix a, Matrix b) {
    return multiply(a, b, false, false);
  }

  /**
   * Multiplies two Matrices and returns the result.
   */
  static Matrix multiply(Matrix a, Matrix b, boolean transposeA, boolean transposeB) {
    Matrix res = new Matrix(transposeA ? a.cols() : a.rows(), transposeB ? b.rows() : b.cols());
    if (transposeA) {
      if (transposeB) {
        if (a.rows() != b.cols())
          throw new IllegalArgumentException("No can do");
        for (int i = 0; i < res.rows(); i++) {
          for (int j = 0; j < res.cols(); j++) {
            double d = 0.0;
            for (int k = 0; k < a.rows(); k++)
              d += a.data.get(k)[i] * b.data.get(j)[k];
            res.data.get(i)[j] = d;
          }
        }
      } else {
        if (a.rows() != b.rows())
          throw new IllegalArgumentException("No can do");
        for (int i = 0; i < res.rows(); i++) {
          for (int j = 0; j < res.cols(); j++) {
            double d = 0.0;
            for (int k = 0; k < a.rows(); k++)
              d += a.data.get(k)[i] * b.data.get(k)[j];
            res.data.get(i)[j] = d;
          }
        }
      }
    } else {
      if (transposeB) {
        if (a.cols() != b.cols())
          throw new IllegalArgumentException("No can do");
        for (int i = 0; i < res.rows(); i++) {
          for (int j = 0; j < res.cols(); j++) {
            double d = 0.0;
            for (int k = 0; k < a.cols(); k++)
              d += a.data.get(i)[k] * b.data.get(j)[k];
            res.data.get(i)[j] = d;
          }
        }
      } else {
        if (a.cols() != b.rows())
          throw new IllegalArgumentException("No can do");
        for (int i = 0; i < res.rows(); i++) {
          for (int j = 0; j < res.cols(); j++) {
            double d = 0.0;
            for (int k = 0; k < a.cols(); k++)
              d += a.data.get(i)[k] * b.data.get(k)[j];
            res.data.get(i)[j] = d;
          }
        }
      }
    }
    return res;
  }

  class SVDResult {
    Matrix u;
    Matrix v;
    double[] diagonal;
  }

  /**
   * Performs singular value decomposition of this matrix
   */
  @SuppressWarnings("ConstantConditions")
  private SVDResult singularValueDecompositionHelper(boolean throwIfNoConverge, int maxIterations) {
    int m = rows();
    int n = cols();
    if (m < n)
      throw new IllegalArgumentException("Expected at least as many rows as columns");
    int j, k;
    int l = 0;
    int p, q;
    double c, f, h, s, x, y, z;
    double norm = 0.0;
    double g = 0.0;
    double scale = 0.0;
    SVDResult res = new SVDResult();
    Matrix pU = new Matrix(m, m);
    res.u = pU;
    pU.fill(0.0);
    for (int i = 0; i < m; i++) {
      double[] rOut = pU.data.get(i);
      double[] rIn = data.get(i);
      for (j = 0; j < n; j++)
        rOut[j] = rIn[j];
    }
    double[] pSigma = new double[n];
    res.diagonal = pSigma;
    Matrix pV = new Matrix(n, n);
    res.v = pV;
    pV.fill(0.0);
    double[] temp = new double[n];

    // Householder reduction to bidiagonal form
    for (int i = 0; i < n; i++) {
      // Left-hand reduction
      temp[i] = scale * g;
      l = i + 1;
      g = 0.0;
      s = 0.0;
      scale = 0.0;
      if (i < m) {
        for (k = i; k < m; k++)
          scale += Math.abs(pU.data.get(k)[i]);
        if (scale != 0.0) {
          for (k = i; k < m; k++) {
            pU.data.get(k)[i] = safeDivide(pU.data.get(k)[i], scale);
            double t = pU.data.get(k)[i];
            s += t * t;
          }
          f = pU.data.get(i)[i];
          g = -takeSign(Math.sqrt(s), f);
          h = f * g - s;
          pU.data.get(i)[i] = f - g;
          if (i != n - 1) {
            for (j = l; j < n; j++) {
              s = 0.0;
              for (k = i; k < m; k++)
                s += pU.data.get(k)[i] * pU.data.get(k)[j];
              f = safeDivide(s, h);
              for (k = i; k < m; k++)
                pU.data.get(k)[j] += f * pU.data.get(k)[i];
            }
          }
          for (k = i; k < m; k++)
            pU.data.get(k)[i] *= scale;
        }
      }
      pSigma[i] = scale * g;

      // Right-hand reduction
      g = 0.0;
      s = 0.0;
      scale = 0.0;
      if (i < m && i != n - 1) {
        for (k = l; k < n; k++)
          scale += Math.abs(pU.data.get(i)[k]);
        if (scale != 0.0) {
          for (k = l; k < n; k++) {
            pU.data.get(i)[k] = safeDivide(pU.data.get(i)[k], scale);
            double t = pU.data.get(i)[k];
            s += t * t;
          }
          f = pU.data.get(i)[l];
          g = -takeSign(Math.sqrt(s), f);
          h = f * g - s;
          pU.data.get(i)[l] = f - g;
          for (k = l; k < n; k++)
            temp[k] = safeDivide(pU.data.get(i)[k], h);
          if (i != m - 1) {
            for (j = l; j < m; j++) {
              s = 0.0;
              for (k = l; k < n; k++)
                s += pU.data.get(j)[k] * pU.data.get(i)[k];
              for (k = l; k < n; k++)
                pU.data.get(j)[k] += s * temp[k];
            }
          }
          for (k = l; k < n; k++)
            pU.data.get(i)[k] *= scale;
        }
      }
      norm = Math.max(norm, Math.abs(pSigma[i]) + Math.abs(temp[i]));
    }

    // Accumulate right-hand transform
    for (int i = n - 1; i >= 0; i--) {
      if (i < n - 1) {
        if (g != 0.0) {
          for (j = l; j < n; j++)
            pV.data.get(i)[j] = safeDivide(safeDivide(pU.data.get(i)[j], pU.data.get(i)[l]), g); // (double-division to avoid underflow)
          for (j = l; j < n; j++) {
            s = 0.0;
            for (k = l; k < n; k++)
              s += pU.data.get(i)[k] * pV.data.get(j)[k];
            for (k = l; k < n; k++)
              pV.data.get(j)[k] += s * pV.data.get(i)[k];
          }
        }
        for (j = l; j < n; j++) {
          pV.data.get(i)[j] = 0.0;
          pV.data.get(j)[i] = 0.0;
        }
      }
      pV.data.get(i)[i] = 1.0;
      g = temp[i];
      l = i;
    }

    // Accumulate left-hand transform
    for (int i = n - 1; i >= 0; i--) {
      l = i + 1;
      g = pSigma[i];
      if (i < n - 1) {
        for (j = l; j < n; j++)
          pU.data.get(i)[j] = 0.0;
      }
      if (g != 0.0) {
        g = safeDivide(1.0, g);
        if (i != n - 1) {
          for (j = l; j < n; j++) {
            s = 0.0;
            for (k = l; k < m; k++)
              s += pU.data.get(k)[i] * pU.data.get(k)[j];
            f = safeDivide(s, pU.data.get(i)[i]) * g;
            for (k = i; k < m; k++)
              pU.data.get(k)[j] += f * pU.data.get(k)[i];
          }
        }
        for (j = i; j < m; j++)
          pU.data.get(j)[i] *= g;
      } else {
        for (j = i; j < m; j++)
          pU.data.get(j)[i] = 0.0;
      }
      pU.data.get(i)[i] += 1.0;
    }

    // Diagonalize the bidiagonal matrix
    for (k = n - 1; k >= 0; k--) // For each singular value
    {
      for (int iteration = 1; iteration <= maxIterations; iteration++) {
        // Test for splitting
        boolean flag = true;
        q = 0;
        for (l = k; l >= 0; l--) {
          q = l - 1;
          if (Math.abs(temp[l]) + norm == norm) {
            flag = false;
            break;
          }
          if (Math.abs(pSigma[q]) + norm == norm)
            break;
        }

        if (flag) {
          c = 0.0;
          s = 1.0;
          for (int i = l; i <= k; i++) {
            f = s * temp[i];
            temp[i] *= c;
            if (Math.abs(f) + norm == norm)
              break;
            g = pSigma[i];
            h = pythag(f, g);
            pSigma[i] = h;
            h = safeDivide(1.0, h);
            c = g * h;
            s = -f * h;
            for (j = 0; j < m; j++) {
              y = pU.data.get(j)[q];
              z = pU.data.get(j)[i];
              pU.data.get(j)[q] = y * c + z * s;
              pU.data.get(j)[i] = z * c - y * s;
            }
          }
        }

        z = pSigma[k];
        if (l == k) {
          // Detect convergence
          if (z < 0.0) {
            // Singular value should be positive
            pSigma[k] = -z;
            for (j = 0; j < n; j++)
              pV.data.get(k)[j] *= -1.0;
          }
          break;
        }
        if (throwIfNoConverge && iteration >= maxIterations)
          throw new IllegalArgumentException("failed to converge");

        // Shift from bottom 2x2 minor
        x = pSigma[l];
        q = k - 1;
        y = pSigma[q];
        g = temp[q];
        h = temp[k];
        f = safeDivide(((y - z) * (y + z) + (g - h) * (g + h)), (2.0 * h * y));
        g = pythag(f, 1.0);
        f = safeDivide(((x - z) * (x + z) + h * (safeDivide(y, (f + takeSign(g, f))) - h)), x);

        // QR transform
        c = 1.0;
        s = 1.0;
        for (j = l; j <= q; j++) {
          int i = j + 1;
          g = temp[i];
          y = pSigma[i];
          h = s * g;
          g = c * g;
          z = pythag(f, h);
          temp[j] = z;
          c = safeDivide(f, z);
          s = safeDivide(h, z);
          f = x * c + g * s;
          g = g * c - x * s;
          h = y * s;
          y = y * c;
          for (p = 0; p < n; p++) {
            x = pV.data.get(j)[p];
            z = pV.data.get(i)[p];
            pV.data.get(j)[p] = x * c + z * s;
            pV.data.get(i)[p] = z * c - x * s;
          }
          z = pythag(f, h);
          pSigma[j] = z;
          if (z != 0.0) {
            z = safeDivide(1.0, z);
            c = f * z;
            s = h * z;
          }
          f = c * g + s * y;
          x = c * y - s * g;
          for (p = 0; p < m; p++) {
            y = pU.data.get(p)[j];
            z = pU.data.get(p)[i];
            pU.data.get(p)[j] = y * c + z * s;
            pU.data.get(p)[i] = z * c - y * s;
          }
        }
        temp[l] = 0.0;
        temp[k] = f;
        pSigma[k] = x;
      }
    }

    // Sort the singular values from largest to smallest
    for (int i = 1; i < n; i++) {
      for (j = i; j > 0; j--) {
        if (pSigma[j - 1] >= pSigma[j])
          break;
        pU.swapColumns(j - 1, j);
        pV.swapRows(j - 1, j);
        double tmp = pSigma[j];
        pSigma[j] = pSigma[j - 1];
        pSigma[j - 1] = tmp;
      }
    }

    // Return results
    pU.fixNans();
    pV.fixNans();
    return res;
  }

  /**
   * Returns the Moore-Penrose pseudoinverse of this matrix
   */
  Matrix pseudoInverse() {
    SVDResult result;
    int columns = cols();
    int rows = rows();
    if (rows < columns) {
      Matrix pTranspose = transpose();
      result = pTranspose.singularValueDecompositionHelper(false, 80);
    } else {
      result = singularValueDecompositionHelper(false, 80);
    }

    Matrix sigma = new Matrix(rows < columns ? columns : rows, rows < columns ? rows : columns);
    sigma.fill(0.0);
    int m = Math.min(rows, columns);
    for (int i = 0; i < m; i++) {
      if (Math.abs(result.diagonal[i]) > 1e-9) {
        sigma.data.get(i)[i] = safeDivide(1.0, result.diagonal[i]);
      } else {
        sigma.data.get(i)[i] = 0.0;
      }
    }

    Matrix pT = Matrix.multiply(result.u, sigma, false, false);
    if (rows < columns) {
      return Matrix.multiply(pT, result.v, false, false);
    } else {
      return Matrix.multiply(result.v, pT, true, true);
    }
  }

  @Override
  public boolean equals(Object object) {
    if (!(object instanceof Matrix)) {
      return false;
    }

    Matrix other = (Matrix) object;
    if (other.rows() != rows() || other.cols() != cols()) {
      return false;
    }

    for (int row = 0; row < rows(); row++) {
      for (int column = 0; column < cols(); column++) {
        if (get(row, column) != get(row, column)) {
          return false;
        }
      }
    }

    return true;
  }

  public static int[] computeFoldSizes(int totalLength, int folds) {
    int foldSize = totalLength / folds;

    int usedRows = foldSize * folds;
    int unusedRows = totalLength - usedRows;

    int[] foldSizes = new int[folds];
    Arrays.fill(foldSizes, foldSize);

    for (int i = 0; i < unusedRows; i++) {
      foldSizes[i] += 1;
    }

    return foldSizes;
  }

  public static Matrix matrixWithoutFold(int[] foldSizes, int foldIndex, Matrix other) {
    int foldSize = foldSizes[foldIndex];

    Matrix newMatrix = new Matrix(other.rows() - foldSize, other.cols());

    int rowsBefore = 0;
    for (int i = 0; i < foldIndex; i++) {
      rowsBefore += foldSizes[i];
    }

    int rowsAfter = other.rows() - foldSize - rowsBefore;

    newMatrix.copyBlock(0, 0, other, 0, 0, rowsBefore, other.cols());
    newMatrix.copyBlock(rowsBefore, 0, other, rowsBefore + foldSize, 0, rowsAfter, other.cols());

    return newMatrix;
  }

  public static Matrix matrixFold(int[] foldSizes, int foldIndex, Matrix other) {
    int foldSize = foldSizes[foldIndex];
    Matrix newMatrix = new Matrix(foldSizes[foldIndex], other.cols());

    int rowsBefore = 0;
    for (int i = 0; i < foldIndex; i++) {
      rowsBefore += foldSizes[i];
    }

    newMatrix.copyBlock(0, 0, other, rowsBefore, 0, foldSize, other.cols());

    return newMatrix;
  }

  public static void shuffleMatrices(Matrix... matrices) {
    if (matrices.length == 0) {
      return;
    }

    final int expectedHeight = matrices[0].rows();
    if (!Arrays.stream(matrices).allMatch((matrix) -> matrix.rows() == expectedHeight)) {
      throw new IllegalArgumentException("All supplied matrices must have the same height.");
    }

    Random random = new Random();
    for (int i = matrices[0].rows(); i >= 2; i--) {
      int r = random.nextInt(i);

      // Swap the same rows in all matrices
      for (Matrix matrix : matrices) {
        matrix.swapRows(i - 1, r);
      }
    }
  }
}
