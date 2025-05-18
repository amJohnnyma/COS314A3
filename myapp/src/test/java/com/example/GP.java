package com.example;

import java.io.IOException;

import org.apache.commons.lang3.ArrayUtils;

import java_cup.reduce_action;

import java.util.*;

import weka.knowledgeflow.Data;

public class GP {
  int MaxDepth = 7;
  int MaxGenerations;
  int PopulationSize;
  long seed;
  Random random;
  ArrayList<Individual> Population;
  double MutationRate = 0.12;

  String[] Arithmetic = { "+", "-", "*", "/", "abs", "log", "sqrt" };
  String[] Comparison = { ">", "<", ">=", "<=", "==" };
  String[] Logical = { "AND", "OR", "NOT" };
  String[] TerminalSet = { "X1", "X2", "X3", "X4", "X5" };

  Dataset trainingSet, testingSet;

  GP(int MaxGenerations, int PopSize) {
    this.MaxGenerations = MaxGenerations;
    this.PopulationSize = PopSize;

    seed = System.currentTimeMillis();
    random = new Random(seed);

    try {
      trainingSet = new Dataset("BTC_train.csv");
      testingSet = new Dataset("BTC_test.csv");
    } catch (IOException e) {
      System.out.println(e.getMessage());
    }
  }

  void Algorithm() {
    ArrayList<Individual> tempPopulation = GenerateInitialPopulation();
    int count = 0;
    while (count < MaxGenerations) {
      CalculateFitness(tempPopulation);
      // get parents
      Individual parentOne = Tournament(tempPopulation);
      Individual parentTwo = Tournament(tempPopulation);
      // create offspring
      Individual[] offspring = crossoverAndMutate(parentOne, parentTwo);
      ArrayList<Individual> OffSpring = new ArrayList<>();
      for (Individual curIndividual : offspring) {
        OffSpring.add(curIndividual);
      }
      CalculateFitness(OffSpring);
      // replace
      tempPopulation = SteadyState(tempPopulation, OffSpring);

      count++;
    }
  }

  ArrayList<Individual> SteadyState(ArrayList<Individual> oldGeneration, ArrayList<Individual> offspring) {
    ArrayList<Individual> newGeneration = new ArrayList<>(oldGeneration);

    for (Individual child : offspring) {
      boolean replaced = false;
      for (int attempts = 0; attempts < 10; attempts++) {
        int index = random.nextInt(newGeneration.size());
        if (child.fitness > newGeneration.get(index).fitness) {
          newGeneration.set(index, child);
          replaced = true;
          break;
        }
      }
    }

    return newGeneration;
  }

  Individual[] crossoverAndMutate(Individual parent1, Individual parent2) {
    // Deep copy both parents
    Individual offspring1 = parent1.clone();
    Individual offspring2 = parent2.clone();

    // Crossover
    performCrossover(offspring1, offspring2);

    // Mutation
    if (Math.random() < MutationRate) {
      mutate(offspring1);
    }
    if (Math.random() < MutationRate) {
      mutate(offspring2);
    }

    return new Individual[] { offspring1, offspring2 };
  }

  void performCrossover(Individual ind1, Individual ind2) {
    Node node1 = getRandomSubtree(ind1.getRoot());
    Node node2 = getRandomSubtree(ind2.getRoot());

    // Swap the subtrees (deep copies for safety)
    Node node1Copy = node1.clone();
    Node node2Copy = node2.clone();

    replaceSubtree(ind1.getRoot(), node1, node2Copy);
    replaceSubtree(ind2.getRoot(), node2, node1Copy);
  }

  void mutate(Individual individual) {
    Node mutationPoint = getRandomSubtree(individual.getRoot());
    Node newSubtree = generateRandomSubtree();
    replaceSubtree(individual.getRoot(), mutationPoint, newSubtree);
  }

  Node generateRandomSubtree() {
    return fullGeneration(3).getRoot();
  }

  Node getRandomSubtree(Node root) {
    List<Node> nodeList = new ArrayList<>();
    collectNodes(root, nodeList);
    return nodeList.get(new Random().nextInt(nodeList.size()));
  }

  void collectNodes(Node node, List<Node> nodes) {
    if (node == null)
      return;
    nodes.add(node);
    if (node instanceof FunctionNode) {
      collectNodes(((FunctionNode) node).getLeftChild(), nodes);
      collectNodes(((FunctionNode) node).getRightChild(), nodes);
    }
  }

  boolean replaceSubtree(Node root, Node target, Node replacement) {
    if (root instanceof FunctionNode) {
      FunctionNode fn = (FunctionNode) root;

      if (fn.getLeftChild() == target) {
        fn.setLeft(replacement);
        return true;
      } else if (fn.getRightChild() == target) {
        fn.setRight(replacement);
        return true;
      } else {
        return replaceSubtree(fn.getLeftChild(), target, replacement)
            || replaceSubtree(fn.getRightChild(), target, replacement);
      }
    }
    return false;
  }

  Individual Tournament(ArrayList<Individual> tempPopulation) {
    Individual a = tempPopulation.get(random.nextInt(tempPopulation.size()));
    Individual b = tempPopulation.get(random.nextInt(tempPopulation.size()));
    return a.fitness > b.fitness ? a : b;
  }

  ArrayList<Individual> GenerateInitialPopulation() {
    // using ramped half-and-half method
    ArrayList<Individual> tempSolution = new ArrayList<>();
    Individual tempIndiv;
    for (int i = 0; i < PopulationSize; i++) {
      if (i < PopulationSize / 2) {
        // create individual using full method
        tempIndiv = fullGeneration(MaxDepth);
      } else {
        // create individual using grow method
        tempIndiv = growGeneration(MaxDepth);
      }
      tempSolution.add(tempIndiv);
    }

    return tempSolution;
  }

  void bindInputs(Node node, double[] input) {
    if (node instanceof TerminalNode) {
      TerminalNode TNode = (TerminalNode) node;
      String label = TNode.label;

      int index = Integer.parseInt(label.substring(1)) - 1;
      TNode.SetValue(input[index]);
    } else {
      FunctionNode FNode = (FunctionNode) node;
      if (FNode.getLeftChild() != null)
        bindInputs(FNode.getLeftChild(), input);
      if (FNode.getRightChild() != null)
        bindInputs(FNode.getRightChild(), input);
    }
  }

  void CalculateFitness(ArrayList<Individual> temp) {
    for (Individual current : temp) {
      int CorrectCount = 0;
      for (int i = 0; i < testingSet.getSize(); i++) {
        double[] inputSet = testingSet.getFeatures(i);
        Integer expectedOutput = testingSet.getExpected(i);

        bindInputs(current.getRoot(), inputSet);

        double result = current.getRoot().evaluate();
        if ((result >= 0.5 && expectedOutput == 1) || (result < 0.5 && expectedOutput == 0)) {
          CorrectCount++;
        }
      }
      current.setFitness(CorrectCount);
    }
  }

  String getRandomFunction() {
    String[] allFunctions = ArrayUtils.addAll(Arithmetic, Comparison);
    allFunctions = ArrayUtils.addAll(allFunctions, Logical);
    return allFunctions[random.nextInt(allFunctions.length)];
  }

  double[] getTerminalSet() {
    int SetSize = trainingSet.getSize();
    return trainingSet.getFeatures(random.nextInt(SetSize));
  }

  Individual fullGeneration(int currentDepth) {
    if (currentDepth >= MaxDepth) {
      return new Individual(new TerminalNode(TerminalSet[random.nextInt(TerminalSet.length)]));
    } else {
      String operator = getRandomFunction();
      Node newNode = new FunctionNode(operator);

      if (operator.equals("NOT") || operator.equals("sqrt") || operator.equals("abs")) {
        newNode.setLeft(fullGeneration(currentDepth + 1).getRoot());
      } else {
        newNode.setLeft(fullGeneration(currentDepth + 1).getRoot());
        newNode.setRight(fullGeneration(currentDepth + 1).getRoot());
      }

      return new Individual(newNode);
    }
  }

  Node getRandomNode(int currentDepth) {
    if (currentDepth >= MaxDepth) {
      return new TerminalNode(TerminalSet[random.nextInt(TerminalSet.length)]);
    } else {
      if (random.nextBoolean()) {
        String operator = getRandomFunction();
        return new FunctionNode(operator);
      } else {
        return new TerminalNode(TerminalSet[random.nextInt(TerminalSet.length)]);
      }
    }
  }

  Individual growGeneration(int currentDepth) {
    Node newNode = getRandomNode(currentDepth);
    if (newNode instanceof FunctionNode) {
      FunctionNode funcNode = (FunctionNode) newNode;
      if (funcNode.getOperator().equals("NOT") || funcNode.getOperator().equals("sqrt")
          || funcNode.getOperator().equals("abs")) {
        funcNode.setLeft(fullGeneration(currentDepth + 1).getRoot());
      } else {
        funcNode.setLeft(fullGeneration(currentDepth + 1).getRoot());
        funcNode.setRight(fullGeneration(currentDepth + 1).getRoot());
      }
    }
    return new Individual(newNode);

  }

}
