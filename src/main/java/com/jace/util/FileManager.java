package com.jace.util;

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

  public static PrintWriter getPrintWriterWithName(String fileName) throws IOException {
    File outputDirectory = getOutputDirectory();
    String filePathName = outputDirectory.toPath().toString() + File.separator + fileName;
    File newFile = new File(filePathName);
    newFile.createNewFile();

    return new PrintWriter(new FileOutputStream(newFile));
  }
}
