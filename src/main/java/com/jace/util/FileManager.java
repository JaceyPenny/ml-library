package com.jace.util;

import com.jace.math.Vector;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

@SuppressWarnings("ResultOfMethodCallIgnored")
public class FileManager {

  private static long EXECUTION_TIMESTAMP = System.currentTimeMillis();

  private static String getOutputDirectoryName() {
    Date date = new Date(EXECUTION_TIMESTAMP);
    SimpleDateFormat fileNameFormat = new SimpleDateFormat("yyyy_MM_dd_hh-mm-ss");
    return fileNameFormat.format(date);
  }

  private static File getOutputDirectory() {
    String directoryName = getOutputDirectoryName();
    File outputDirectory = new File("output" + File.separator + directoryName);

    outputDirectory.mkdirs();
    return outputDirectory;
  }

  public static File getOutputFileWithName(String fileName) throws IOException {
    File outputDirectory = getOutputDirectory();
    String filePathName = outputDirectory.toPath().toString() + File.separator + fileName;
    File newFile = new File(filePathName);
    newFile.createNewFile();

    return newFile;
  }

  public static PrintWriter getPrintWriterWithName(String fileName) throws IOException {
    File outputFile = getOutputFileWithName(fileName);

    return new PrintWriter(new FileOutputStream(outputFile));
  }

  private static int getPixel(double r, double g, double b) {
    return getPixel((int) r, (int) g, (int) b);
  }

  private static int getPixel(int r, int g, int b) {
    return (r << 16) + (g << 8) + b;
  }

  public static void writeImageFromVector(Vector image, int width, int height) throws IOException {
    writeImageFromVector(image, width, height, false);
  }

  public static void writeImageFromVector(Vector image, int width, int height, boolean isRGB)
      throws IOException {
    Vector imageVector = image;
    if (!isRGB) {
      imageVector = image.map((value) -> (double) Math.round(value * 256));
    }

    BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    for (int w = 0; w < width; w++) {
      for (int h = 0; h < height; h++) {
        int position = (h * width + w) * 3;
        double r = imageVector.get(position);
        double g = imageVector.get(position + 1);
        double b = imageVector.get(position + 2);

        bufferedImage.setRGB(w, h, getPixel(r, g, b));
      }
    }

    File outputFile = getOutputFileWithName("output-image.png");
    ImageIO.write(bufferedImage, "PNG", outputFile);
  }
}
