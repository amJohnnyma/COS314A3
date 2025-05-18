package com.example;

import org.netlib.util.doubleW;

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

  double getValue() {
    return Value;
  }

}
