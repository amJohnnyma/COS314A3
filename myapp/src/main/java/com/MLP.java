package com;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;




class Layer //input, hidden, output //currently not used //probably doesnt need to exist
{
    public Vector<Neuron> neurons = new Vector<Neuron>();
}

public class MLP {
    

    private List<double[]> inputsVal = new ArrayList<>(); ///full input of each
    private List<double[]> inputs = new ArrayList<>(); //5 inputs per // one input per feature (Open, high, low, close, adj close)
    private List<String> inputsTime = new ArrayList<>(); //Orderof each 5 inputs//might be useless
    private List<Double> labels = new ArrayList<>(); //expected output (last input)
    private Neuron outputNeuron;
    private Neuron[] hiddenLayer;
    private int numSamples;

    //file to read, how many hidden layers, how many inputs (5)
    public MLP(String file, int hiddenSize, int inputSize, int seed) //constructor
    {
        //read data and assign inputs

    // Open,High,Low,Close,Adj Close,Output
    // -1.104550546,-1.103639854,-1.103311839,-1.100806056,-1.460681595,0
    //inputsval = numbers
    //inputstime = time of day

        String csvFile = "src/data/BTC_train.csv";
        String line;
        String csvSplitBy = ",";            

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] values = line.split(csvSplitBy);

                if(!containsNumbers(line))
                {
                    if(inputsTime.size() <= 0) //redundant check
                    {
                        for(int k = 0; k < values.length; k++)
                        {
                            inputsTime.add(values[k]);
                        }
                    }
                }
                else{
                    numSamples++;
                    double[] row = new double[6];
                    for(int k =0; k < values.length; k++)
                    {
                        row[k] = Double.parseDouble(values[k]);
                    }
                    inputsVal.add(row);

                }
                
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        //get into useable form
        for(double[] row : inputsVal)
        {
            double[] input = Arrays.copyOfRange(row, 0, 5);
            double label = row[5];

            inputs.add(input);
            labels.add(label);

        }

        hiddenLayer = new Neuron[hiddenSize];
        for(int k = 0; k < hiddenSize;k++)
        {
            hiddenLayer[k] = new Neuron(inputSize, seed);
        }

        outputNeuron = new Neuron(hiddenSize, seed);


    }

            /*
         * 
        TRAINING PHASE
        ---------------
        input (x_train) -> feedForward -> loss -> backpropagation -> update weights
        pseudo code
        for (epoch = 0; epoch < numEpochs; epoch++) {
            for (int i = 0; i < trainInputs.length; i++) {
                double[] predicted = feedForward(trainInputs[i]);
                double loss = lossFunction(trainLabels[i], predicted);
                backpropagate(trainLabels[i], predicted); // updates weights
            }
        }

        TESTING PHASE
        --------------
        input (x_test) -> feedForward -> compare prediction to y_test (NO weight update)
        pseudo code
        int correct = 0;
        for (int i = 0; i < testInputs.length; i++) {
            double[] predicted = feedForward(testInputs[i]); // use trained weights
            int predictedLabel = predicted[0] >= 0.5 ? 1 : 0;
            if (predictedLabel == testLabels[i]) {
                correct++;
            }
        }
        System.out.println("Accuracy: " + (correct * 100.0 / testInputs.length) + "%");



         */

    public void trainNetwork(int iterations)
    {
        for(int k = 0; k < iterations; k ++)
        {
            for (int i = 0; i < inputsVal.size(); i++) {
            double[] input = inputs.get(i);
            double expectedOutput = labels.get(i);

            double[] predictedOutput = feedForward(input);

            double lf = lossFunction(input, expectedOutput, predictedOutput);

            System.out.println("Expected: " + expectedOutput + " Predicted: " + predictedOutput[0]);
            System.out.println("Loss: " + lf);
            //back prop
            }
        }

    }

    public void testSolution()
    {

    }
   
    ////////////Helper functions// Could probably be its own class
    public static boolean containsNumbers(String input) {
        Pattern pattern = Pattern.compile("\\d");
        Matcher matcher = pattern.matcher(input);
        return matcher.find();
    }

    public double[] feedForward(double[] input)
    {
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activate(input);
        }

        double output = outputNeuron.activate(hiddenOutputs);
        return new double[] { output };
        
    }

    //not sure if i must use predictedOutput.length or numSamples
    public double lossFunction(double[] inp, double expected, double[] predictedOutput)
    {
        // THIS IS NOT A REGRESSION PROBLEM THIS FORMULA IS WRONG
        // SHOULD USE CROSS-ENTROPY
        // https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/
        /*
         L=−1/N ​∑i=1 to N​[y.i​ * log(y^​i​) + (1–y.i​) * log(1–y^​i​)]

        Where:

            y.i​ is the actual label.
            y^i​ is the predicted label.
            N is the number of samples.

            //needs gaurd against log(0)
            double e = 1e-7; // log(max(predicted[i], e))
            double loss = 0.0;
            for(i to predictedOutput.length)
            {
                loss += expected * log(predicted[i]) + (1-expected) * log(1-predicted[i]);
            }

            return  -loss / predicted.length;

         */
        double sum = 0.0;
        for(int i = 0; i < predictedOutput.length; i ++)
        {
            sum += Math.pow((expected - predictedOutput[i]),2);
        }
        double mse = sum / predictedOutput.length;

        return mse;
    }
}
