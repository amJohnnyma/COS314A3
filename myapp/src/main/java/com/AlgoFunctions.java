package com;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AlgoFunctions {

    public void runMLP(int iterations, int runs, double learningRate, long seed, int batchSize, int hiddenSize, int hiddenLayers, double targetAccuracy) {
        boolean useRandomSeeds = true;
        if(seed != 0)
        {
            useRandomSeeds = false;
        }

        int numRuns = 0;
        if(!useRandomSeeds)
        {
            numRuns = 1; //no need for more runs since fixed seed
        }
        else{
            numRuns = runs;
        }
        Random r = new Random();

        // Create a thread pool with N threads (adjust based on CPU cores)
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads - 2);

        for (int k = 0; k < numRuns; k++) {

            final String chartName = "TRAINING_Batch_" + batchSize + "_HS_" + hiddenSize
                    + "_LR_" + learningRate + "_Seed_" + seed;
            final long s;
            if(useRandomSeeds)
            {
                s = r.nextLong();
            }
            else{
                s = seed;
            }

            executor.submit(() -> {
                try {
                    MLP mlp = new MLP("src/data/BTC_train.csv", hiddenSize, hiddenLayers, 5, s,
                            learningRate);
                    mlp.trainNetwork(iterations, batchSize, 50, 0.01);

                    double accuracy = mlp.testNetwork();
                    if (accuracy >= targetAccuracy) {
                        // Prepare a data holder class or map
                        TrainingMetrics data = new TrainingMetrics(
                                mlp.getLosses(),
                                mlp.avgWeights,
                                mlp.avgBiases,
                                mlp.epochTimes,
                                mlp.deltaValues);

                        data.saveRawData(chartName + ".json");

                        // Save the model as usual
                        mlp.saveModel(accuracy + " : " + chartName + ".mlp");
                        System.out.println("Finished: " + chartName);

                        //mlp.testModel


                    }
                } catch (Exception e) {
                    System.err.println("Failed Training: " + chartName);
                    e.printStackTrace();
                }


            });
        }

        // Shutdown the executor and wait for all tasks to finish
        executor.shutdown();
    }

}
