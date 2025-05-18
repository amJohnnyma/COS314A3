package com.example;

import java.util.*;

import org.netlib.util.doubleW;

abstract class Node {
  Node left, right;
  void setLeft(Node child){
    this.left = child;
  }

  void setRight(Node child){
    this.right = child;
  }

  Node getLeftChild(){
    return left;
  }

  Node getRightChild(){
    return right;
  }

  abstract double evaluate();
  public abstract Node clone();
}

