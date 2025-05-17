package com;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

//smoothing tool

/*
WithStops/_Batch_16_HS_16_LR_0.05_Seed_-208296179699223446.json
WithStops/_Batch_16_HS_32_LR_0.05_Seed_4553753751109754545.json
WithStops/_Batch_16_HS_16_LR_0.05_Seed_-3121642471003736982.json
WithStops/_Batch_32_HS_16_LR_0.1_Seed_-8652245330602223685.json
WithStops/_Batch_16_HS_16_LR_0.05_Seed_4032348048418809032.json   
WithStops/_Batch_32_HS_32_LR_0.01_Seed_-2413647171961181879.json
WithStops/_Batch_16_HS_16_LR_0.05_Seed_-8581540863905620641.json  
WithStops/_Batch_32_HS_32_LR_0.01_Seed_5444187905547035713.json
WithStops/_Batch_16_HS_16_LR_0.1_Seed_8969997661819792942.json    
WithStops/_Batch_32_HS_32_LR_0.05_Seed_8164454498107071600.json
WithStops/_Batch_16_HS_32_LR_0.01_Seed_-9192080552531894582.json
 */

public class Main {
    public static void main(String[] args) {

    //    double[] lr = { 0.1, 0.05, 0.01 };
        // long seed = 0;
        Random r = new Random();

        // Create a thread pool with N threads (adjust based on CPU cores)
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads - 2);

                    for (int k = 0; k < 20; k++) {
                        final long seed = r.nextLong();
                        final int it = 5000; 
                        final int batch = 16; 
                        final int hiddenSize = 32; 
                        final int hiddenLayers = 2; 
                        final double learningRate = 0.01;
                        final String chartName = "_Batch_" + batch + "_HS_" + hiddenSize
                                + "_LR_" + learningRate + "_Seed_" + seed;

                        executor.submit(() -> {
                            try {
                                MLP mlp = new MLP("src/data/BTC_train.csv", hiddenSize, hiddenLayers, 5, seed,
                                        learningRate);
                                mlp.trainNetwork(it, batch, 50, 0.01);

                                double accuracy = mlp.testNetwork();
                                if (accuracy >= 0.95) 
                                {
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
