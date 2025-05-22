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
    return root.toString();
  }

  public Integer getSize(){
    return root.getSize();
  }

  private void collectTerminalValues(Node node, Map<String, Double> values) {
    if (node instanceof TerminalNode) {
      TerminalNode tn = (TerminalNode) node;
      values.put(tn.label, tn.Value);
    } else if (node instanceof FunctionNode) {
      FunctionNode fn = (FunctionNode) node;
      if (fn.getLeftChild() != null)
        collectTerminalValues(fn.getLeftChild(), values);
      if (fn.getRightChild() != null)
        collectTerminalValues(fn.getRightChild(), values);
    }
  }

  private String buildExpression(Node node) {
    if (node instanceof TerminalNode) {
      return ((TerminalNode) node).label;
    } else if (node instanceof FunctionNode) {
      FunctionNode fn = (FunctionNode) node;
      String operator = fn.getOperator();

      switch (operator) {
        case "NOT":
        case "sqrt":
        case "abs":
        case "log":
          // Unary operators
          return operator + "(" + buildExpression(fn.getLeftChild()) + ")";
        default:
          // Binary operators
          return "(" + buildExpression(fn.getLeftChild()) + " " + operator + " " + buildExpression(fn.getRightChild())
              + ")";
      }
    }
    return "";
  }

}
