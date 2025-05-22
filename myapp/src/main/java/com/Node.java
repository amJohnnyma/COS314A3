package com;

abstract class Node {
  Node left, right;

  void setLeft(Node child) {
    this.left = child;
  }

  void setRight(Node child) {
    this.right = child;
  }

  Node getLeftChild() {
    return left;
  }

  Node getRightChild() {
    return right;
  }

  abstract double evaluate();

  public abstract Node clone();

  public abstract String toString();

  public abstract Integer getSize();

  abstract String toTreeString(String indent);

  public String toPrettyString() {
    return toPrettyString("", true);
  }

  protected String toPrettyString(String prefix, boolean isTail) {
    StringBuilder sb = new StringBuilder();
    sb.append(prefix).append(isTail ? "└── " : "├── ").append(toString()).append("\n");

    if (left != null && right != null) {
      sb.append(left.toPrettyString(prefix + (isTail ? "    " : "│   "), false));
      sb.append(right.toPrettyString(prefix + (isTail ? "    " : "│   "), true));
    } else if (left != null) {
      sb.append(left.toPrettyString(prefix + (isTail ? "    " : "│   "), true));
    }

    return sb.toString();
  }
}
