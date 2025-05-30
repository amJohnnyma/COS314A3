package com;

import java.util.Random;
import java.util.Scanner;
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
        String answer;
        Scanner scanner = new Scanner(System.in);
        System.out.println("Run MLP or GP. Enter 1 for MLP and 2 for GP");
        answer = scanner.nextLine();
        if (answer.equals("1")) {
            // Dewald main code
            // double[] lr = { 0.1, 0.05, 0.01 };
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

                        // double accuracy = mlp.testNetwork();
                        // if (accuracy >= 0.95)
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
                            // mlp.saveModel(accuracy + " : " + chartName + ".mlp");
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
        } else if (answer.equals("2")) {
            // Herrie Main code
            GP gp = new GP(100, 20);
            gp.Algorithm();
            Individual besIndividual = gp.getBestIndividual();
            System.out.println();
            System.out.println(besIndividual.toString());
            System.out.println("Fitness of best individual: "+ besIndividual.fitness);
            System.out.println("Accuracy of best individual: "+ gp.GetAccuracyOfBestIndividual());
        }

    }

}
/*

public class Main 
{
    public static void main( String[] args )
    {
        System.out.println("=== Cryptocurrency Stock Predictor ===\n");
        
        // Example 1: Use default random seed (different each run)
        System.out.println("Example 1: Default random seed");
        CryptoStockPredictor predictor1 = new CryptoStockPredictor();
        predictor1.DT();
        
        System.out.println("\n=== Test Complete ===\n");
        
        // Example 2: Generate new random seed for another run
        System.out.println("\nExample 2: New random seed for same instance");
        predictor1.randomizeSeed();
        predictor1.DT();
        
        System.out.println("\n=== Test Complete ===\n");
        
        // Example 3: Use specific seed for reproducible results
        System.out.println("\nExample 3: Fixed seed for reproducible results");
        CryptoStockPredictor predictor2 = new CryptoStockPredictor(42);
        predictor2.DT();
        
        System.out.println("\n=== Test Complete ===\n");
        
        // Example 4: Switch between random and fixed seed
        System.out.println("\nExample 4: Switch to random seed mode");
        predictor2.setUseRandomSeed(true); // This will generate a new random seed
        predictor2.DT();
        
        System.out.println("\n=== All Tests Complete ===");
        
        // Uncomment to test with unpruned tree
        // predictor1.setUnpruned(true);
        // predictor1.DT();
    }
}
*/
