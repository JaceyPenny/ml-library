package com.jace.evaluator;

public class MetricTracker {
  private int steps;
  private long startTime;
  private long totalTime;


  public MetricTracker() {
    startTime = System.nanoTime();
  }

  public long getSteps() {
    return steps;
  }

  public long getTime() {
    return totalTime;
  }

  public void start() {
    startTime = System.nanoTime();
  }

  public void pause() {
    totalTime += System.nanoTime() - startTime;
  }

  public void updateSteps() {
    updateSteps(1);
  }

  public void updateSteps(int amount) {
    steps += amount;
  }

  public void reset() {
    startTime = System.nanoTime();
    steps = 0;
    totalTime = 0;
  }
}
