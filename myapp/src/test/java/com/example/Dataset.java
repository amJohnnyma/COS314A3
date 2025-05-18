package com.example;

import java.io.*;
import java.util.*;

public class Dataset {
  public List<double[]> features = new ArrayList<>();
  public List<Integer> labels = new ArrayList<>();

  public Dataset(String filename) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(filename));
    String line = reader.readLine();
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(",");
      double[] input = new double[tokens.length - 1];
      for (int i = 0; i < input.length; i++) {
        input[i] = Double.parseDouble(tokens[i]);
      }
      int label = Integer.parseInt(tokens[tokens.length - 1]);
      features.add(input);
      labels.add(label);
    }
    reader.close();
  }

  int getSize(){
    return features.size();
  }

  double[] getFeatures(int index){
    return features.get(index);
  }

  Integer getExpected(int index){
    return labels.get(index);
  }

}
