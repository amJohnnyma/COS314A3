package com.example;

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
    // First collect all terminal values
    StringBuilder variables = new StringBuilder();
    Map<String, Double> terminalValues = new LinkedHashMap<>();
    collectTerminalValues(root, terminalValues);

    for (Map.Entry<String, Double> entry : terminalValues.entrySet()) {
      if (variables.length() > 0)
        variables.append(", ");
      variables.append(entry.getKey()).append("=").append(String.format("%.2f", entry.getValue()));
    }

    // Then build the expression
    String expression = buildExpression(root);

    return variables.toString() + " then " + expression;
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
