package com.example;

public class Individual {
  Node root;
  double fitness;

  public Individual(Node root) {
    this.root = root;
    this.fitness = 0;
  }

  void setFitness(double fitness){
    this.fitness = fitness;
  }

  Node getRoot(){
    return root;
  }

  public Individual clone() {
    return new Individual(root.clone());
  }



}
