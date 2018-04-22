package com.jace.util;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Scanner;

public class Console {

  private static MessageLevel messageLevel = MessageLevel.WARNING;

  private static SimpleDateFormat formatter = new SimpleDateFormat("dd MMM, hh:mm:ssa");
  private static Scanner scanner = new Scanner(System.in);

  private static boolean isAcceptingInput = true;

  private static String fiftySpaces = "                                        ";

  private static boolean lastMessageWasProgress = false;

  public enum MessageLevel {
    ERROR, INFO, WARNING, DEBUG
  }

  public static void init() {
    System.out.print("ML Library '18\n> ");
  }

  public static void setMessageLevel(MessageLevel level) {
    messageLevel = level;
  }

  public static void out(MessageLevel type, String message, Object... args) {
    if (lastMessageWasProgress) {
      System.out.print("\r> ");
      lastMessageWasProgress = false;
    }

    System.out.printf(
        "%s | %s: %s\n> ",
        formatter.format(new Date()),
        type.toString(),
        String.format(message, args)
    );
  }

  public static void disableInput() {
    isAcceptingInput = false;
  }

  public static void enableInput() {
    isAcceptingInput = true;
  }

  public static String getLine() {
    String result = scanner.nextLine();
    System.out.print("> ");
    return result;
  }

  public static String get() {
    String result = scanner.next();
    System.out.print("> ");
    return result;
  }

  public static int getInt() {
    int result = scanner.nextInt();
    System.out.print("> ");
    return result;
  }

  public static double getDouble() {
    double result = scanner.nextDouble();
    System.out.print("> ");
    return result;
  }

  public static long getLong() {
    long result = scanner.nextLong();
    System.out.print("> ");
    return result;
  }

  public static boolean getBoolean() {
    boolean result = scanner.nextBoolean();
    System.out.print("> ");
    return result;
  }

  public static float getFloat() {
    float result = scanner.nextFloat();
    System.out.print("> ");
    return result;
  }

  public static boolean hasNext() {
    return scanner.hasNext();
  }

  public static void d(String message, Object... args) {
    if (messageLevel == MessageLevel.DEBUG) {
      out(MessageLevel.DEBUG, message, args);
    }
  }

  public static void w(String message, Object... args) {
    if (messageLevel == MessageLevel.DEBUG || messageLevel == MessageLevel.WARNING) {
      out(MessageLevel.WARNING, message, args);
    }
  }

  public static void i(String message, Object... args) {
    if (messageLevel == MessageLevel.DEBUG
        || messageLevel == MessageLevel.WARNING
        || messageLevel == MessageLevel.INFO) {
      out(MessageLevel.INFO, message, args);
    }
  }

  public static void e(String message, Object... args) {
    out(MessageLevel.ERROR, message, args);
  }

  public static void exception(Exception e) {
    e(e.getClass().getCanonicalName() + ": " + e.getMessage());

    for (StackTraceElement el : e.getStackTrace()) {
      e("\t" + el.toString());
    }
  }

  public static void progress(String message, double percentage) {
    lastMessageWasProgress = true;
    System.out.print("\r> ");
    System.out.printf(
        "%s | %s: %s: %.1f%%",
        formatter.format(new Date()),
        MessageLevel.INFO,
        message,
        percentage
    );
    System.out.print(fiftySpaces);
  }
}