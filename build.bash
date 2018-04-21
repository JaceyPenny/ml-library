#!/bin/bash

echo "Compiling..."
javac -d classes -cp ./src/main/java ./src/main/java/com/jace/Main.java
echo
echo "To run:"
echo "    $ java -cp classes com.jace.Main"
echo
