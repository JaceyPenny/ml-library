package com.jace.math;

import com.jace.util.Json;

import java.util.ArrayList;
import java.util.HashMap;

public class Metadata {
  public static final double UNKNOWN_VALUE = -1e308;

  private String fileName;                          // the name of the file
  private ArrayList<String> attributeNames;                 // the name of each attribute (or column)
  private ArrayList<HashMap<String, Integer>> attributeToValueMaps; // value to enumeration
  private ArrayList<HashMap<Integer, String>> valueToAttributeMaps; // enumeration to value

  public Metadata() {
    fileName = "";
    attributeNames = new ArrayList<>();
    attributeToValueMaps = new ArrayList<>();
    valueToAttributeMaps = new ArrayList<>();
  }

  public Metadata(String fileName) {
    this();
    this.fileName = fileName;
  }

  public Metadata(Metadata other) {
    this.fileName = other.fileName;
    this.attributeNames = new ArrayList<>();
    this.attributeNames.addAll(other.attributeNames);
    this.attributeToValueMaps.addAll(other.attributeToValueMaps);
    this.valueToAttributeMaps.addAll(other.valueToAttributeMaps);
  }

  public String getFileName() {
    return fileName;
  }

  public void setFileName(String fileName) {
    this.fileName = fileName;
  }

  public ArrayList<String> getAttributeNames() {
    return attributeNames;
  }

  public Metadata copy() {
    return new Metadata(this);
  }

  public void clear() {
    fileName = "";
    attributeNames.clear();
    attributeToValueMaps.clear();
    valueToAttributeMaps.clear();
  }

  public void addColumn(String name) {
    attributeNames.add(name);
    attributeToValueMaps.add(new HashMap<>());
    valueToAttributeMaps.add(new HashMap<>());
  }

  public void addColumn(int valueCount) {
    String name = "col_" + attributeNames.size();

    attributeNames.add(name);

    HashMap<String, Integer> attributeToValueMap = new HashMap<>();
    HashMap<Integer, String> valueToAttributeMap = new HashMap<>();

    for (int i = 0; i < valueCount; i++) {
      String sVal = "val_" + i;
      attributeToValueMap.put(sVal, i);
      valueToAttributeMap.put(i, sVal);
    }

    attributeToValueMaps.add(attributeToValueMap);
    valueToAttributeMaps.add(valueToAttributeMap);
  }

  public void copyAttributeToMetadata(Metadata other, int sourceAttribute, int destinationAttribute) {
    other.attributeNames.set(destinationAttribute, attributeNames.get(sourceAttribute));
    other.attributeToValueMaps.set(
        destinationAttribute, new HashMap<>(attributeToValueMaps.get(sourceAttribute)));
    other.valueToAttributeMaps.set(
        destinationAttribute, new HashMap<>(valueToAttributeMaps.get(destinationAttribute)));
  }

  public void swapColumns(int a, int b) {
    String attributeNameTemp = attributeNames.get(a);
    attributeNames.set(a, attributeNames.get(b));
    attributeNames.set(b, attributeNameTemp);

    HashMap<String, Integer> attributeToValueMapTemp = attributeToValueMaps.get(a);
    attributeToValueMaps.set(a, attributeToValueMaps.get(b));
    attributeToValueMaps.set(b, attributeToValueMapTemp);

    HashMap<Integer, String> valueToAttributeMapTemp = valueToAttributeMaps.get(a);
    valueToAttributeMaps.set(a, valueToAttributeMaps.get(b));
    valueToAttributeMaps.set(b, valueToAttributeMapTemp);
  }

  public Integer getValueForAttributeInColumn(String attribute, int column) {
    if (column < 0 || column >= attributeToValueMaps.size()) {
      throw new IllegalArgumentException("Index out of bounds: " + column);
    } else if (!attributeToValueMaps.get(column).containsKey(attribute)) {
      throw new IllegalArgumentException("No attribute named: " + attribute);
    }

    return attributeToValueMaps.get(column).get(attribute);
  }

  public Integer findOrCreateValueForAttributeInColumn(String attribute, int column) {
    if (column < 0 || column >= attributeToValueMaps.size()) {
      throw new IllegalArgumentException("Index out of bounds: " + column);
    }

    if (!attributeToValueMaps.get(column).containsKey(attribute)) {
      int newValue = attributeToValueMaps.get(column).size();
      attributeToValueMaps.get(column).put(attribute, newValue);
      valueToAttributeMaps.get(column).put(newValue, attribute);
      return newValue;
    } else {
      return getValueForAttributeInColumn(attribute, column);
    }
  }

  public String getAttributeForValueInColumn(int value, int column) {
    if (column < 0 || column >= attributeToValueMaps.size()) {
      throw new IllegalArgumentException("Index out of bounds: " + column);
    } else if (!valueToAttributeMaps.get(column).containsKey(value)) {
      throw new IllegalArgumentException(
          "No value \"" + value + "\" in enum \"" + attributeNames.get(column) + "\"");
    }

    return valueToAttributeMaps.get(column).get(value);
  }

  public int getEnumSizeForAttributeInColumn(int column) {
    if (column < 0 || column >= attributeToValueMaps.size()) {
      throw new IllegalArgumentException("Index out of bounds: " + column);
    }

    return valueToAttributeMaps.get(column).size();
  }

  public boolean attributeIsContinuous(int column) {
    return getEnumSizeForAttributeInColumn(column) == 0;
  }

  public HashMap<String, Integer> getAttributeToValueMapForColumn(int column) {
    if (column < 0 || column >= attributeToValueMaps.size()) {
      throw new IllegalArgumentException("Index out of bounds: " + column);
    }

    return attributeToValueMaps.get(column);
  }

  public HashMap<Integer, String> getValueToAttributeMapForColumn(int column) {
    if (column < 0 || column >= attributeToValueMaps.size()) {
      throw new IllegalArgumentException("Index out of bounds: " + column);
    }

    return valueToAttributeMaps.get(column);
  }

  public void parseAttribute(String line) {
    HashMap<String, Integer> attributeToValueMap = new HashMap<>();
    HashMap<Integer, String> valueToAttributeMap = new HashMap<>();
    attributeToValueMaps.add(attributeToValueMap);
    valueToAttributeMaps.add(valueToAttributeMap);

    Json.StringParser stringParser = new Json.StringParser(line);
    stringParser.advance(10);
    stringParser.skipWhitespace();
    stringParser.advance(1);
    String attributeName = stringParser.until('\'');
    stringParser.advance(1);

    attributeNames.add(attributeName);
    stringParser.skipWhitespace();

    int valueCount = 0;
    if (stringParser.peek() == '{') {  // Start of enumeration
      stringParser.advance(1);
      while (stringParser.peek() != '}') {
        stringParser.skipWhitespace();
        String value = stringParser.untilQuoteSensitive(',', '}');

        if (stringParser.peek() == ',') {
          stringParser.advance(1);
        }

        if (attributeToValueMap.containsKey(value)) {
          throw new RuntimeException("Duplicate attribute value: " + value);
        }

        attributeToValueMap.put(value, valueCount);
        valueToAttributeMap.put(valueCount, value);
        valueCount++;
      }

      stringParser.advance(1);
    }
  }
}
