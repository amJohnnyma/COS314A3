package com;

public class TerminalNode extends Node {
  double Value;
  String label;

  public TerminalNode(String label) {
    this.label = label;
  }

  public TerminalNode(String label, Node left, Node right) {
    this.label = label;
    this.left = left;
    this.right = right;
  }

  void SetValue(double val) {
    Value = val;
  }

  @Override
  double evaluate() {
    return Value;
  }

  @Override
  public Node clone() {
    TerminalNode clone = new TerminalNode(this.label);
    clone.SetValue(this.Value);
    return clone;
  }

  @Override
  public String toString() {
    return label + ": " + String.format("%.6f", Value);
  }

  @Override
  public Integer getSize() {
    return 1;
  }

  @Override
  public String toTreeString(String indent) {
    return indent + "Terminal: " + label + " = " + Value + "\n";
  }

  double getValue() {
    return Value;
  }

}
