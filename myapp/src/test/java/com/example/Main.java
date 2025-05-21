package com.example;

import java.util.Collections;
import java.util.Comparator;

public class Main {
  public static void main(String[] args){
    GP gp = new GP(10, 6);
    gp.Algorithm();
    Individual besIndividual = gp.getBestIndividual();
    System.out.println(besIndividual.toString());
  }
}
