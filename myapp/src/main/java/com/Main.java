package com;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main
{
    public static void main( String[] args )
    {



  double[] lr = {0.001, 0.0005, 0.0001, 0.005};
       // long seed = 0;
        Random r = new Random();
        final long seed = r.nextLong();

        // Create a thread pool with N threads (adjust based on CPU cores)
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

       //     for (int j = 1; j <= 10; j*=2) {
          //       for (int hs = 8; hs <= 16; hs *= 2) {
               //      for (int hl = 1; hl <= 8; hl+=2) {
              //          for (int l = 0; l < lr.length; l++) {

                            final int it = 300; //300 - 1000 //500 feels good
                            final int batch = 1; //to 64 //1 or it just guesses
                            final int hiddenSize = 32; //to 32 //16 pretty good
                            final int hiddenLayers = 3; //to 3
                            final double learningRate = 0.01;// 0.001 to 0.01 //0.05 seems like the sweet spot
                            final String chartName = "Test_100:It_" + it + "_Batch_" + batch + "_HS_" + hiddenSize + "_HL_" + hiddenLayers + "_LR_" + learningRate + "_Seed_" + seed;

                            executor.submit(() -> {
                                try {
                                    MLP mlp = new MLP("src/data/Test.csv", hiddenSize, hiddenLayers, 5, seed, learningRate);
                                    mlp.trainNetwork(it, batch, 30, 0.00001);

                                    Graph g = new Graph(
                                        mlp.getLosses(),
                                        mlp.avgWeights,
                                        mlp.avgBiases,
                                        mlp.epochTimes,
                                        mlp.deltaValues
                                    );
                                    mlp.testNetwork();
                                    g.createAndShowChart(chartName + ".png");
                                //    mlp.saveModel(chartName + ".mlp"); //No saving now. Dont need a saved trained model

                                    System.out.println("Finished: " + chartName);
                                } catch (Exception e) {
                                    System.err.println("Failed: " + chartName);
                                    e.printStackTrace();
                                }
                            });
                //          }
                //      }
                // }
       //     }
        

        // Shutdown the executor and wait for all tasks to finish
        executor.shutdown();
    }
      


}
