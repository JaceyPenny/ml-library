package com.jace.util;

import com.jace.math.Matrix;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ChartMaker {

  private Color backgroundColor = Color.WHITE;
  private Color axisColor = new Color(128, 128, 128);
  private Color gridColor = new Color(192, 192, 192);
  private Color labelColor = Color.BLACK;
  private Color pointBorderColor = Color.WHITE;
  private Color pointColor = Color.BLUE;

  private boolean connectPoints = false;

  private double x_min, x_max;
  private double y_min, y_max;

  private double bufferPercentage;

  private int width, height;

  private double pointSize = 5;

  private double labelFontSize = 12;

  private Matrix data;

  private BufferedImage image;
  private Graphics2D graphics;

  public ChartMaker() {
  }

  public void setConnectPoints(boolean connectPoints) {
    this.connectPoints = connectPoints;
  }

  public void setWidth(int width) {
    this.width = width;
  }

  public void setHeight(int height) {
    this.height = height;
  }

  public void setDimensions(int width, int height) {
    this.width = width;
    this.height = height;
  }

  public void setChartRange(double x_min, double x_max, double y_min, double y_max) {
    this.x_min = x_min;
    this.x_max = x_max;
    this.y_min = y_min;
    this.y_max = y_max;
  }

  public void setBufferPercentage(double bufferPercentage) {
    this.bufferPercentage = bufferPercentage;
  }

  public void setPointSize(double pointSize) {
    this.pointSize = pointSize;
  }

  public void setData(Matrix data) {
    this.data = data;
  }

  public void setLabelFontSize(double labelFontSize) {
    this.labelFontSize = labelFontSize;
  }

  private void initialize() throws IllegalStateException {
    if (x_min == 0 && x_max == 0 && y_min == 0 && y_max == 0) {
      x_min = data.columnMin(0);
      x_max = data.columnMax(0);

      y_min = data.columnMin(1);
      y_max = data.columnMax(1);

      double x_dist = Math.abs(x_max - x_min);
      double y_dist = Math.abs(y_max - y_min);

      x_min -= bufferPercentage * x_dist;
      x_max += bufferPercentage * x_dist;
      y_min -= bufferPercentage * y_dist;
      y_max += bufferPercentage * y_dist;
    }

    if (width <= 0 && height <= 0) {
      throw new IllegalStateException("You must specify a width and height for the output image");
    } else if (width <= 0 || height <= 0) {
      double x_dist = Math.abs(x_max - x_min);
      double y_dist = Math.abs(y_max - y_min);
      double aspectRatio = x_dist / y_dist;

      if (width > 0) {
        height = (int) Math.round(width / aspectRatio);
      } else {
        width = (int) Math.round(height * aspectRatio);
      }
    }

    image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    graphics = image.createGraphics();

    graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
  }

  private double map(double value, double inMin, double inMax, double outMin, double outMax) {
    return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
  }

  private double xInImage(double xInChart) {
    return map(xInChart, x_min, x_max, 0, width);
  }

  private double yInImage(double yInChart) {
    return map(yInChart, y_min, y_max, height, 0);
  }

  private Point2D pointInImage(Point2D pointInChart) {
    Point2D result = new Point2D.Double();
    double x = xInImage(pointInChart.getX());
    double y = yInImage(pointInChart.getY());

    result.setLocation(x, y);
    return result;
  }

  public void draw() {
    initialize();

    // fill background with
    graphics.setColor(Color.WHITE);
    graphics.fillRect(0, 0, width, height);

    // Calculate axis information
    Point2D x_axis_start = pointInImage(new Point2D.Double(x_min, 0));
    Point2D x_axis_end = pointInImage(new Point2D.Double(x_max, 0));
    Point2D y_axis_start = pointInImage(new Point2D.Double(0, y_max));
    Point2D y_axis_end = pointInImage(new Point2D.Double(0, y_min));

    // draw grid lines
    graphics.setColor(gridColor);
    graphics.setStroke(new BasicStroke(2.0f));

    for (int x = (int) Math.ceil(x_min); x <= Math.floor(x_max); x++) {
      if (x == 0) {
        continue;
      }

      double imageX = xInImage(x);
      drawLine(imageX, y_axis_start.getY(), imageX, y_axis_end.getY());
    }

    for (int y = (int) Math.ceil(y_min); y <= Math.floor(y_max); y++) {
      if (y == 0) {
        continue;
      }

      double imageY = yInImage(y);
      drawLine(x_axis_start.getX(), imageY, x_axis_end.getX(), imageY);
    }

    // draw axes
    graphics.setColor(axisColor);
    graphics.setStroke(new BasicStroke(3.0f));

    graphics.drawLine((int) x_axis_start.getX(), (int) x_axis_start.getY(),
        (int) x_axis_end.getX(), (int) x_axis_end.getY());
    graphics.drawLine((int) y_axis_start.getX(), (int) y_axis_start.getY(),
        (int) y_axis_end.getX(), (int) y_axis_end.getY());

    pointColor = Color.getHSBColor(0, 1, 1);
    if (connectPoints) {
      // draw the point connections
      graphics.setStroke(new BasicStroke(2.0f));

      double xLast = xInImage(data.row(0).get(0));
      double yLast = yInImage(data.row(0).get(1));

      for (int i = 1; i < data.rows(); i++) {
        pointColor = Color.getHSBColor((float) i / data.rows(), 1, 1);
        graphics.setColor(pointColor);

        double imageX = xInImage(data.row(i).get(0));
        double imageY = yInImage(data.row(i).get(1));

        drawLine(xLast, yLast, imageX, imageY);

        xLast = imageX;
        yLast = imageY;
      }
    }

    // draw the points
    pointColor = Color.getHSBColor(0, 1, 1);
    graphics.setStroke(new BasicStroke(1.0f));

    for (int i = 0; i < data.rows(); i++) {
      pointColor = Color.getHSBColor((float) i / data.rows(), 1, 1);

      double imageX = xInImage(data.row(i).get(0));
      double imageY = yInImage(data.row(i).get(1));

      drawPoint(imageX, imageY);
    }

    // draw integer labels
    graphics.setFont(new Font("TimesRoman", Font.PLAIN, (int) labelFontSize));
    graphics.setColor(labelColor);

    for (int x = (int) Math.ceil(x_min); x <= Math.floor(x_max); x++) {
      double imageX = xInImage(x);
      drawText(Integer.toString(x), imageX + 2, height - 4);
    }

    for (int y = (int) Math.ceil(y_min); y <= Math.floor(y_max); y++) {
      double imageY = yInImage(y);
      drawText(Integer.toString(y), 4, imageY - 4);
    }
  }

  private void drawLine(double x1, double y1, double x2, double y2) {
    graphics.drawLine((int) x1, (int) y1, (int) x2, (int) y2);
  }

  private void drawPoint(double x, double y) {
    double ovalStartX = x - pointSize / 2;
    double ovalStartY = y - pointSize / 2;
    graphics.setColor(pointColor);
    graphics.fillOval((int) ovalStartX, (int) ovalStartY, (int) pointSize, (int) pointSize);
    graphics.setColor(pointBorderColor);
    graphics.drawOval((int) ovalStartX, (int) ovalStartY, (int) pointSize, (int) pointSize);
  }

  private void drawText(String text, double x, double y) {
    graphics.drawString(text, (int) x, (int) y);
  }

  public void writeToFile(File file) throws IOException {
    ImageIO.write(image, "PNG", file);
  }
}
