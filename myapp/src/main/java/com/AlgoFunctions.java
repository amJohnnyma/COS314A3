package com;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.TimeUnit;


public class AlgoFunctions {

    public boolean runMLP(int iterations, int runs, double learningRate, long seed, int batchSize, int hiddenSize, int hiddenLayers, double targetAccuracy) {
    AtomicInteger completed = new AtomicInteger(0);  // Shared thread-safe counter

        boolean useRandomSeeds = (seed == 0);
        final int numRuns;
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
        ExecutorService executor = Executors.newFixedThreadPool(numThreads-1);

        for (int k = 0; k < numRuns; k++) {

            final long s;
            if(useRandomSeeds)
            {
                s = r.nextLong();
            }
            else{
                s = seed;
            }

                        final int it = k;
            final String chartName = "TRAINING_Batch_" + batchSize + "_HS_" + hiddenSize
                    + "_LR_" + learningRate + "_Seed_" + s;
//System.out.println("Training..." + k);
            executor.submit(() -> {
                try {
                 //   System.out.println("Thread handled");
                  //  System.out.println(System.getProperty("user.dir") +"/myapp/"+ "src/data/BTC_train.csv");
                    MLP mlp = new MLP(System.getProperty("user.dir") +"/myapp/"+ "src/data/BTC_train.csv", hiddenSize, hiddenLayers, 5, s,
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
                int progress = completed.incrementAndGet();
        System.out.println("Iteration " + progress + " / " + numRuns);

            });
        }

try {
    executor.shutdown();
    return executor.awaitTermination(1, TimeUnit.HOURS);
} catch (InterruptedException e) {
    e.printStackTrace();
    return false;
}

    }

    public void testMLP(String filename)
    {
        try
        {
           // System.out.println(filename);
            MLP mlp = MLP.loadModel(System.getProperty("user.dir") + "/" + filename);
         //   System.out.println("Load model");
            double accuracy = mlp.testNetwork();
            System.out.println("Accuracy " + accuracy);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }

}
