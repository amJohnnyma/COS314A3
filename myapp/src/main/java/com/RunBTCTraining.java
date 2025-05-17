package com;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class RunBTCTraining {
    public static void main(String[] args) {
        System.out.println("Starting Bitcoin price prediction training...");
        
        // Training parameters
        final int iterations = 5000;
        final int batchSize = 16;
        final int hiddenSize = 32;
        final int hiddenLayers = 2;
        final double learningRate = 0.01;
        final double targetAccuracy = 0.95;
        
        // Create a fixed seed for reproducibility (or use Random for variability)
        final long seed = 12345; // Fixed seed for reproducibility
        
        // Number of parallel training sessions
        final int numRuns = 10; 
        
        System.out.println("Configuration:");
        System.out.println("- Iterations: " + iterations);
        System.out.println("- Batch size: " + batchSize);
        System.out.println("- Hidden size: " + hiddenSize);
        System.out.println("- Hidden layers: " + hiddenLayers);
        System.out.println("- Learning rate: " + learningRate);
        System.out.println("- Target accuracy: " + targetAccuracy);
        System.out.println("- Seed: " + seed);
        System.out.println("- Number of runs: " + numRuns);
        
        // Set up executor service
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads - 2);
        System.out.println("Using " + (numThreads - 2) + " threads");
        
        for (int k = 0; k < numRuns; k++) {
            final String chartName = "BTC_Training_Batch_" + batchSize + "_HS_" + hiddenSize
                    + "_LR_" + learningRate + "_Seed_" + seed + "_Run" + k;
            
            System.out.println("Submitting training job: " + chartName);
            
            executor.submit(() -> {
                try {
                    System.out.println("Starting training: " + chartName);
                    long startTime = System.currentTimeMillis();
                    
                    // Initialize MLP with BTC_train.csv
                    MLP mlp = new MLP("src/data/BTC_train.csv", hiddenSize, hiddenLayers, 5, seed,
                            learningRate);
                    
                    // Train the network
                    mlp.trainNetwork(iterations, batchSize, 50, 0.01);
                    
                    // Test the network
                    double accuracy = mlp.testNetwork();
                    System.out.println("Training " + chartName + " completed with accuracy: " + accuracy);
                    
                    // Save regardless of accuracy for analysis
                    // Prepare data for visualization
                    TrainingMetrics data = new TrainingMetrics(
                            mlp.getLosses(),
                            mlp.avgWeights,
                            mlp.avgBiases,
                            mlp.epochTimes,
                            mlp.deltaValues);
                    
                    // Save raw training data
                    data.saveRawData(chartName + ".json");
                    
                    // Save the model
                    mlp.saveModel(accuracy + " : " + chartName + ".mlp");
                    
                    long endTime = System.currentTimeMillis();
                    System.out.println("Training time: " + ((endTime - startTime) / 1000) + " seconds");
                    System.out.println("Finished: " + chartName);
                    
                    // Create visualization
                    try {
                        Graph graph = new Graph(mlp.getLosses(), mlp.avgWeights, mlp.avgBiases, 
                                                mlp.epochTimes.stream().map(t -> (double)t/1000).toList(), 
                                                mlp.deltaValues);
                        graph.createChart(chartName);
                        System.out.println("Created chart for: " + chartName);
                    } catch (Exception e) {
                        System.err.println("Failed to create chart: " + e.getMessage());
                    }
                    
                } catch (Exception e) {
                    System.err.println("Failed Training: " + chartName);
                    e.printStackTrace();
                }
            });
        }
        
        // Shutdown the executor and wait for all tasks to finish
        executor.shutdown();
        
        try {
            // Wait for all tasks to complete
            System.out.println("Waiting for all training tasks to complete...");
            if (!executor.awaitTermination(2, TimeUnit.HOURS)) {
                System.err.println("Training tasks didn't complete within timeout");
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            System.err.println("Training was interrupted");
            executor.shutdownNow();
        }
        
        System.out.println("All training tasks completed!");
    }
}