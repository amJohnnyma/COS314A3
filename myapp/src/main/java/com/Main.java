package com;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main
{
    public static void main( String[] args )
    {
        /*
        //HS: 8-32 HL: 1-2 LR: 0.0001 - 0.00001
        //Only test different iterations with different batch size for now to avoid blowing up my pc
        double[] lr = {0.0005, 0.001, 0.005, 0.01, 0.05};
        long seed = 0;
     //   int k = 10;
     //   int j = 1;
     ////   int hl = 1;
      //  int hs = 16;
      //  int l = 0;
        for(int k = 1; k <= 5; k ++)
        {
            for(int j = 1; j <= 5;j++)
            {
                for(int hs = 16; hs < 128; hs*=2)
                {
                    for(int hl = 1; hl <= 5; hl++)
                    {
                        for(int l = 0; l < 5; l++)
                        {
                        MLP mlp = new MLP("src/data/BTC_train.csv", hs, hl, 5, seed, lr[l]); 
                        mlp.trainNetwork(100 * k, 8 * j);
                        Graph g = new Graph(
                            mlp.getLosses(),
                            mlp.avgWeights,
                            mlp.avgBiases,
                            mlp.epochTimes
                        );
                        String chartName = "It_" + (k * 100) + "_Batch_" + (8 * j) + "_HS_" + hs + "_HL_" + hl + "_LR_" + lr[l];
                        g.createAndShowChart(chartName + ".png");
                        mlp.saveModel(chartName + ".mlp");                        
                        }

                    }

                }

            }
        }
  */


  double[] lr = {0.001, 0.0005, 0.0001, 0.005};
        int seed = 0;

        // Create a thread pool with N threads (adjust based on CPU cores)
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

            for (int j = 1; j <= 5; j++) {
                for (int hs = 16; hs < 128; hs *= 2) {
                    for (int hl = 1; hl <= 4; hl++) {
                        for (int l = 0; l < lr.length; l++) {

                            final int it = 500;
                            final int batch = 8 * j;
                            final int hiddenSize = hs;
                            final int hiddenLayers = hl;
                            final double learningRate = lr[l];
                            final String chartName = "It_" + it + "_Batch_" + batch + "_HS_" + hs + "_HL_" + hl + "_LR_" + learningRate;

                            executor.submit(() -> {
                                try {
                                    MLP mlp = new MLP("src/data/BTC_train.csv", hiddenSize, hiddenLayers, 5, seed, learningRate);
                                    mlp.trainNetwork(it, batch);

                                    Graph g = new Graph(
                                        mlp.getLosses(),
                                        mlp.avgWeights,
                                        mlp.avgBiases,
                                        mlp.epochTimes
                                    );
                                    g.createAndShowChart(chartName + ".png");
                                    mlp.saveModel(chartName + ".mlp");

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
