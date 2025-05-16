package com;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main {
    public static void main(String[] args) {

        double[] lr = { 0.1, 0.05, 0.01, 0.005 };
        // long seed = 0;
        Random r = new Random();

        // Create a thread pool with N threads (adjust based on CPU cores)
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        for (int ler = 0; ler < 4; ler++) {
            for (int hs = 1; hs <= 2; hs++) {
                for (int bs = 1; bs <= 2; bs++) {
                    for (int k = 0; k < 10; k++) {
                        final long seed = r.nextLong();
                        final int it = 5000; // 300 - 1000 //500 feels good
                        final int batch = 16 * bs; // to 64 //1 or it just guesses
                        final int hiddenSize = 16 * hs; // to 32 //16 pretty good
                        final int hiddenLayers = 2; // to 3
                        final double learningRate = lr[ler];// 0.001 to 0.01 //0.05 seems like the sweet spot
                        final String chartName = "_Batch_" + batch + "_HS_" + hiddenSize
                                + "_LR_" + learningRate + "_Seed_" + seed;

                        executor.submit(() -> {
                            try {
                                MLP mlp = new MLP("src/data/BTC_train.csv", hiddenSize, hiddenLayers, 5, seed,
                                        learningRate);
                                mlp.trainNetwork(it, batch, 50, 1e-4);

                                double accuracy = mlp.testNetwork();
                                if (accuracy >= 0.95) {

                                    Graph g = new Graph(
                                            mlp.getLosses(),
                                            mlp.avgWeights,
                                            mlp.avgBiases,
                                            mlp.epochTimes,
                                            mlp.deltaValues);
                                    g.createAndShowChart(accuracy + " : " + chartName + ".png");
                                    mlp.saveModel(accuracy + " : " + chartName + ".mlp");
                                }

                                System.out.println("Finished: " + chartName);
                            } catch (Exception e) {
                                System.err.println("Failed: " + chartName);
                                e.printStackTrace();
                            }
                        });
                    }
                }
            }

        }

        // Shutdown the executor and wait for all tasks to finish
        executor.shutdown();
    }

}
