package com;

import java.util.LinkedHashMap;
import java.util.Map;

public class Individual {
  Node root;
  double fitness;

  public Individual(Node root) {
    this.root = root;
    this.fitness = 0;
  }

  void setFitness(double fitness) {
    this.fitness = fitness;
  }

  Node getRoot() {
    return root;
  }

  public Individual clone() {
    return new Individual(root.clone());
  }

  public String toString() {
    return root.toPrettyString();
  }

  public String toTreeString() {
    return root.toTreeString("");
  }

  public Integer getSize() {
    return root.getSize();
  }

}
