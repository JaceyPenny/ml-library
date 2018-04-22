package com.jace.util;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Scanner;

public class Console {

  private static LogType mCurrentLogLevel = LogType.WARNING;

  private static SimpleDateFormat mFormatter = new SimpleDateFormat("dd MMM, hh:mm:ssa");
  private static Scanner mScanner = new Scanner(System.in);

  private static boolean isAcceptingInput = true;

  private static String mFiftySpaces = "                                        ";

  private static boolean mProgressLast = false;

  public enum LogType {
    ERROR, INFO, WARNING, DEBUG
  }

  public static void init() {
    System.out.print("ML Library '18\n> ");
  }

  public static void out(LogType type, String message, Object... args) {
    if (mProgressLast) {
      System.out.print("\r> ");
      mProgressLast = false;
    }

    System.out.printf(
        "%s | %s: %s\n> ",
        mFormatter.format(new Date()),
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
    String result = mScanner.nextLine();
    System.out.print("> ");
    return result;
  }

  public static String get() {
    String result = mScanner.next();
    System.out.print("> ");
    return result;
  }

  public static int getInt() {
    int result = mScanner.nextInt();
    System.out.print("> ");
    return result;
  }

  public static double getDouble() {
    double result = mScanner.nextDouble();
    System.out.print("> ");
    return result;
  }

  public static long getLong() {
    long result = mScanner.nextLong();
    System.out.print("> ");
    return result;
  }

  public static boolean getBoolean() {
    boolean result = mScanner.nextBoolean();
    System.out.print("> ");
    return result;
  }

  public static float getFloat() {
    float result = mScanner.nextFloat();
    System.out.print("> ");
    return result;
  }

  public static boolean hasNext() {
    return mScanner.hasNext();
  }

  public static void d(String message, Object... args) {
    if (mCurrentLogLevel == LogType.DEBUG) {
      out(LogType.DEBUG, message, args);
    }
  }

  public static void w(String message, Object... args) {
    if (mCurrentLogLevel == LogType.DEBUG || mCurrentLogLevel == LogType.WARNING) {
      out(LogType.WARNING, message, args);
    }
  }

  public static void i(String message, Object... args) {
    if (mCurrentLogLevel == LogType.DEBUG
        || mCurrentLogLevel == LogType.WARNING
        || mCurrentLogLevel == LogType.INFO) {
      out(LogType.INFO, message, args);
    }
  }

  public static void e(String message, Object... args) {
    out(LogType.ERROR, message, args);
  }

  public static void exception(Exception e) {
    e(e.getClass().getCanonicalName() + ": " + e.getMessage());

    for (StackTraceElement el : e.getStackTrace()) {
      e("\t" + el.toString());
    }
  }

  public static void progress(String message, double percentage) {
    mProgressLast = true;
    System.out.print("\r> ");
    System.out.printf(
        "%s | %s: %s: %.1f%%",
        mFormatter.format(new Date()),
        LogType.INFO,
        message,
        percentage
    );
    System.out.print(mFiftySpaces);
  }
}