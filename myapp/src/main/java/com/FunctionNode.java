package com;

public class FunctionNode extends Node {
  String operator;

  public FunctionNode(String operator) {
    this.operator = operator;
    this.left = null;
    this.right = null;
  }

  public FunctionNode(String operator, Node left, Node right) {
    this.operator = operator;
    this.left = left;
    this.right = right;
  }

  @Override
  double evaluate() {
    double leftVal = (left != null) ? left.evaluate() : 0;
    double rightVal = (right != null) ? right.evaluate() : 0;

    switch (operator) {
      // Arithmetic
      case "+":
        return leftVal + rightVal;
      case "-":
        return leftVal - rightVal;
      case "*":
        return leftVal * rightVal;
      case "/":
        return (rightVal == 0) ? 1 : leftVal / rightVal;
      case "abs":
        return Math.abs(leftVal);
      case "log":
        return (leftVal <= 0) ? 0 : Math.log(leftVal);
      case "sqrt":
        return (leftVal < 0) ? 0 : Math.sqrt(leftVal);

      // Comparison (return 1.0 for true, 0.0 for false)
      case ">":
        return (leftVal > rightVal) ? 1.0 : 0.0;
      case "<":
        return (leftVal < rightVal) ? 1.0 : 0.0;
      case ">=":
        return (leftVal >= rightVal) ? 1.0 : 0.0;
      case "<=":
        return (leftVal <= rightVal) ? 1.0 : 0.0;
      case "==":
        return (leftVal == rightVal) ? 1.0 : 0.0;

      // Logical (treat non-zero as true)
      case "AND":
        return ((leftVal != 0) && (rightVal != 0)) ? 1.0 : 0.0;
      case "OR":
        return ((leftVal != 0) || (rightVal != 0)) ? 1.0 : 0.0;
      case "NOT":
        return (leftVal == 0) ? 1.0 : 0.0;
      default:
        throw new RuntimeException("Unknown operator " + operator);
    }
  }

  @Override
  public Node clone() {
    Node leftClone = (left != null) ? left.clone() : null;
    Node rightClone = (right != null) ? right.clone() : null;
    return new FunctionNode(operator, leftClone, rightClone);
  }

  @Override
  public String toString() {
    return operator;
  }

  @Override
  public Integer getSize() {
    int size = 1;
    if (left != null) {
      size += left.getSize();
    }
    if (right != null) {
      size += right.getSize();
    }
    return size;
  }

  @Override
  public String toTreeString(String indent) {
    StringBuilder sb = new StringBuilder();
    sb.append(indent).append("Function: ").append(operator).append("\n");

    if (left != null)
      sb.append(left.toTreeString(indent + "  "));
    if (right != null)
      sb.append(right.toTreeString(indent + "  "));

    return sb.toString();
  }

  void setLeft(Node child) {
    this.left = child;
  }

  void setRight(Node child) {
    this.right = child;
  }

  String getOperator() {
    return operator;
  }

  Node getLeftChild() {
    return left;
  }

  Node getRightChild() {
    return right;
  }

}
