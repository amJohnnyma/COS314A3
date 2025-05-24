package com;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.TimeUnit;

public class AlgoFunctions {

    public List<String> runMLP(int iterations, int runs, double learningRate, long seed, int batchSize, int hiddenSize,
            int hiddenLayers, double targetAccuracy, int patience, double minImpro) {
        AtomicInteger completed = new AtomicInteger(0); // Shared thread-safe counter

        boolean useRandomSeeds = (seed == 0);
        final int numRuns;
        if (!useRandomSeeds) {
            numRuns = 1; // no need for more runs since fixed seed
        } else {
            numRuns = runs;
        }
        Random r = new Random();

        // Create a thread pool with N threads (adjust based on CPU cores)
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads - 1);
List<String> successfulRuns = Collections.synchronizedList(new ArrayList<>());
        List<Future<String>> futures = new ArrayList<>();

        for (int k = 0; k < numRuns; k++) {

            final long s;
            if (useRandomSeeds) {
                s = r.nextLong();
            } else {
                s = seed;
            }

            final String chartName = "TRAINING_Batch_" + batchSize + "_HS_" + hiddenSize
                    + "_LR_" + learningRate + "_Seed_" + s;
            // System.out.println("Training..." + k);
            futures.add(executor.submit(() -> {
                try {
                    // System.out.println("Thread handled");
                    // System.out.println(System.getProperty("user.dir") +"/myapp/"+
                    // "src/data/BTC_train.csv");
                    MLP mlp = new MLP(System.getProperty("user.dir") + "/myapp/" + "src/data/BTC_train.csv", hiddenSize,
                            hiddenLayers, 5, s,
                            learningRate);
                    mlp.trainNetwork(iterations, batchSize, patience, minImpro);

                    String[] accuracy = mlp.testNetwork();
                    if (Double.parseDouble(accuracy[0]) >= targetAccuracy) {
                        // Prepare a data holder class or map
                        TrainingMetrics data = new TrainingMetrics(
                                mlp.getLosses(),
                                mlp.avgWeights,
                                mlp.avgBiases,
                                mlp.epochTimes,
                                mlp.deltaValues);

                        data.saveRawData(chartName + ".json");

                        // Save the model as usual
                        mlp.saveModel(accuracy[0] + " : " + chartName + ".mlp");
                        System.out.println("Finished: " + chartName);

                        // mlp.testModel
                        successfulRuns.add(accuracy[1]);

                    }
                } catch (Exception e) {
                    System.err.println("Failed Training: " + chartName);
                    e.printStackTrace();
                }
                int progress = completed.incrementAndGet();
                System.out.println("Iteration " + progress + " / " + numRuns);
                return new String("Iteration " + progress + " / " + numRuns);
            }));
        }

        executor.shutdown();
        try {
            executor.awaitTermination(1, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // Return all collected successful results
        return successfulRuns;

    }

    public String testMLP(String filename) {
        try {
            // System.out.println(filename);
            MLP mlp = MLP.loadModel(System.getProperty("user.dir") + "/" + filename);
            // System.out.println("Load model");
            String results = mlp.networkRWTest();
            return results;
          //  System.out.println("Accuracy " + accuracy);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "";
    }

}
