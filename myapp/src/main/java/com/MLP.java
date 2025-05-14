package com;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

class Layer // input, hidden, output //currently not used //probably doesnt need to exist
{
    public Vector<Neuron> neurons = new Vector<Neuron>();
}

public class MLP {

    private List<double[]> inputsVal = new ArrayList<>(); /// full input of each
    private List<double[]> inputs = new ArrayList<>(); // 5 inputs per // one input per feature (Open, high, low, close,
                                                       // adj close)
    private List<String> inputsTime = new ArrayList<>(); // Orderof each 5 inputs//might be useless
    private List<Double> labels = new ArrayList<>(); // expected output (last input)
    private Neuron outputNeuron; // one per label
    private Neuron[] hiddenLayer;
    private int numSamples = 0;
    private double learningRate = 0.0;

    //Data collection
    private Vector<Double> losses = new Vector<Double>();
    private Vector<Double> avgGradient = new Vector<Double>();
    private Vector<Double> expectedValues = new Vector<Double>();
    private Vector<Double> predictedValues = new Vector<Double>();
    private Vector<Double> accuracyOverTime = new Vector<Double>();
    private Vector<Double> recallOverTime = new Vector<Double>();
    private Vector<Double> f1ScoreOverTime = new Vector<Double>();
    


    ////

    // file to read, how many hidden layers, how many inputs (5)
    public MLP(String file, int hiddenSize, int inputSize, long seed, double learningRate) // constructor
    {
        // read data and assign inputs

        // Open,High,Low,Close,Adj Close,Output
        // -1.104550546,-1.103639854,-1.103311839,-1.100806056,-1.460681595,0
        // inputsval = numbers
        // inputstime = time of day
        this.learningRate = learningRate;

        String csvFile = file;
        String line;
        String csvSplitBy = ",";

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] values = line.split(csvSplitBy);

                if (!containsNumbers(line)) {
                    if (inputsTime.size() <= 0) // redundant check
                    {
                        for (int k = 0; k < values.length; k++) {
                            inputsTime.add(values[k]);
                        }
                    }
                } else {
                    numSamples++;
                    double[] row = new double[6];
                    for (int k = 0; k < values.length; k++) {
                        row[k] = Double.parseDouble(values[k]);
                    }
                    inputsVal.add(row);

                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // get into useable form
        for (double[] row : inputsVal) {
            double[] input = Arrays.copyOfRange(row, 0, 5);
            double label = row[5];

            inputs.add(input);
            labels.add(label);

        }

        hiddenLayer = new Neuron[hiddenSize];
        for (int k = 0; k < hiddenSize; k++) {
            hiddenLayer[k] = new Neuron(inputSize, seed);
        }

        outputNeuron = new Neuron(hiddenSize, seed);

    }

    /*
     * 
     * TRAINING PHASE
     * ---------------
     * input (x_train) -> feedForward -> loss -> backpropagation -> update weights
     * pseudo code
     * for (epoch = 0; epoch < numEpochs; epoch++) {
     * for (int i = 0; i < trainInputs.length; i++) {
     * double[] predicted = feedForward(trainInputs[i]);
     * double loss = lossFunction(trainLabels[i], predicted);
     * backpropagate(trainLabels[i], predicted); // updates weights
     * }
     * }
     * 
     * TESTING PHASE
     * --------------
     * input (x_test) -> feedForward -> compare prediction to y_test (NO weight
     * update)
     * pseudo code
     * int correct = 0;
     * for (int i = 0; i < testInputs.length; i++) {
     * double[] predicted = feedForward(testInputs[i]); // use trained weights
     * int predictedLabel = predicted[0] >= 0.5 ? 1 : 0;
     * if (predictedLabel == testLabels[i]) {
     * correct++;
     * }
     * }
     * System.out.println("Accuracy: " + (correct * 100.0 / testInputs.length) +
     * "%");
     * 
     * 
     * 
     */

    public void trainNetwork(int iterations) {// https://chatgpt.com/share/6824badd-a01c-8012-bb91-21d7c211a6a0
        for (int k = 0; k < iterations; k++) {
            for (int i = 0; i < inputs.size(); i++) {
                double[] input = inputs.get(i);
                double expectedOutput = labels.get(i);

                System.out.println("Input size: " + input.length);
                double[] predictedOutput = feedForward(input);

                double lf = lossFunction(input, expectedOutput, predictedOutput);

                //used for graphing
                losses.add(lf);

                expectedValues.add(expectedOutput);
                predictedValues.add(predictedOutput[0]);

            //    System.out.println("Expected: " + expectedOutput + " Predicted: " + predictedOutput[0]);
            //    System.out.println("Loss: " + lf);

                // 1. gradient calculation
                /*
                 * A gradient is a derivative — it tells you how fast something is changing.

                 * //Will use Batch gradient descent since small data set
                 */

                // gradient at hidden layer
                double delta = predictedOutput[0] - expectedOutput;
                // update weights of output neuron
                double avgGrad = 0.0;
                for (int j = 0; j < hiddenLayer.length; j++) {
                    double gradient = delta * hiddenLayer[j].output; // dL/dW = delta * activation
                    outputNeuron.updateWeights(learningRate, gradient);
                    avgGrad += gradient;
                }                
                avgGrad /= hiddenLayer.length;
                avgGradient.add(avgGrad);
                // update bias
                outputNeuron.bias -= learningRate * delta;

                // 2. error propagation
                // propagate back (chain rule) hidden layer
                double[] deltaHidden = new double[hiddenLayer.length];
                for (int j = 0; j < hiddenLayer.length; j++) {
                    deltaHidden[j] = delta * outputNeuron.weights[j] * hiddenLayer[j].output
                            * (1 - hiddenLayer[j].output);
                }

                // 3. gradient descent
                // update weights of hidden neurons
                for (int j = 0; j < hiddenLayer.length; j++) {
                    for (int in = 0; in < input.length; in++) {
                        double dw = deltaHidden[j] * input[in];
                        hiddenLayer[j].updateWeights(learningRate, dw);
                    }
                    hiddenLayer[j].bias -= learningRate * deltaHidden[j];
                }
            }

        }

    }

    public void testSolution() {

    }

    //////////// Helper functions// Could probably be its own class
    public static boolean containsNumbers(String input) {
        Pattern pattern = Pattern.compile("\\d");
        Matcher matcher = pattern.matcher(input);
        return matcher.find();
    }

    public double[] feedForward(double[] input) {
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activate(input);
        }

        double output = outputNeuron.activate(hiddenOutputs);
        return new double[] { output };

    }

    // not sure if i must use predictedOutput.length or numSamples
    public double lossFunction(double[] inp, double expected, double[] predictedOutput) {

        double e = 1e-7; // to gaurd against log(0)
        double loss = 0.0;

        for (int i = 0; i < predictedOutput.length; i++) {
            loss += expected * Math.log(Math.max(e, predictedOutput[i]))
                    + (1 - expected) * Math.log(Math.max(e, 1 - predictedOutput[i]));
        }

        return -loss / predictedOutput.length;


    }
}
