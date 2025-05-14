package com;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

class Layer {
    Neuron[] neurons;

    public Layer(int inputSize, int numNeurons, long seed) {
        neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(inputSize, seed);
        }
    }

    public double[] forward(double[] input) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].activate(input);
        }
        return outputs;
    }
}

public class MLP {

    private List<double[]> inputsVal = new ArrayList<>(); /// full input of each
    private List<double[]> inputs = new ArrayList<>(); // 5 inputs per // one input per feature (Open, high, low, close,
    private List<Layer> hiddenLayers = new ArrayList<>();
    private List<String> inputsTime = new ArrayList<>(); // Orderof each 5 inputs//might be useless
    private List<Double> labels = new ArrayList<>(); // expected output (last input)
    private Neuron outputNeuron; // one per label
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

    public MLP(String file, int hiddenSize, int hiddenL, int inputSize, long seed, double learningRate) // constructor
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


        int currentInputSize = inputSize;
        for(int i =0; i < hiddenL; i++)
        {
            Layer layer = new Layer(currentInputSize, hiddenSize,seed);
            hiddenLayers.add(layer);
            currentInputSize = hiddenSize;
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
    
        for(int k = 0; k < iterations; k++) //epochs
        {
            for(int i = 0; i < inputs.size(); i++)            
            {
                double[] input = inputs.get(i);
                double[] out = input;
                double expectedOutput = labels.get(i);
                for(Layer layer : hiddenLayers)
                {
                    out = layer.forward(out); //feedforward each neuron in the layer
                }
                double prediction = outputNeuron.activate(out); //final prediction

                double lf = lossFunction(out, expectedOutput, prediction);
                losses.add(lf);
                 System.out.println("Expected: " + expectedOutput + " Predicted: " + prediction);
                System.out.println("Loss: " + lf);

                backward(out, expectedOutput, prediction);

            } 
        }
    
    
        /*   for (int k = 0; k < iterations; k++) {
            for (int i = 0; i < inputs.size(); i++) {
                double[] input = inputs.get(i);
                double expectedOutput = labels.get(i);

             //   System.out.println("Input size: " + input.length);
                double[] predictedOutput = feedForward(input);

                double lf = lossFunction(input, expectedOutput, predictedOutput);

                //used for graphing
                losses.add(lf);

                expectedValues.add(expectedOutput);
                predictedValues.add(predictedOutput[0]);

            //    System.out.println("Expected: " + expectedOutput + " Predicted: " + predictedOutput[0]);
            //    System.out.println("Loss: " + lf);

                // 1. gradient calculation


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
*/
    }

    public void testSolution() {

    }

    //////////// Helper functions// Could probably be its own class
    public static boolean containsNumbers(String input) {
        Pattern pattern = Pattern.compile("\\d");
        Matcher matcher = pattern.matcher(input);
        return matcher.find();
    }

    public void backward(double[] input, double expectedOutput, double prediction)
    {
            // Step 1: Compute the error (delta) at the output layer
    double delta = prediction - expectedOutput;
    
    // Update output neuron weights (outputNeuron is the final layer neuron)
    for (int i = 0; i < hiddenLayers.get(hiddenLayers.size() - 1).neurons.length; i++) {
        double gradient = delta * hiddenLayers.get(hiddenLayers.size() - 1).neurons[i].output;
        outputNeuron.updateWeights(learningRate, gradient);
    }

    



    // Update output neuron bias
    outputNeuron.bias -= learningRate * delta;

    // Step 2: Backpropagate the error through hidden layers
    double[] deltaHidden = new double[hiddenLayers.get(hiddenLayers.size() - 1).neurons.length];
    
    // Loop through hidden layers backwards
    for (int l = hiddenLayers.size() - 1; l >= 0; l--) {
        Layer layer = hiddenLayers.get(l);
        double[] newDeltaHidden = new double[layer.neurons.length];
        
        // Calculate delta for each neuron in this layer
        for (int j = 0; j < layer.neurons.length; j++) {
            Neuron neuron = layer.neurons[j];
            double out = neuron.output;
            
            // Backpropagate delta through each neuron
            for (int k = 0; k < neuron.weights.length; k++) {
                double inputToThisNeuron = neuron.input[k]; // saved during forward pass
                double gradient = deltaHidden[j] * inputToThisNeuron;
                neuron.weights[k] -= learningRate * gradient;
            }

            // Update the neuron bias
            neuron.bias -= learningRate * deltaHidden[j];

            // Propagate delta to the next layer (if not the first hidden layer)
            if (l > 0) {
                for (int k = 0; k < hiddenLayers.get(l - 1).neurons.length; k++) {
                    newDeltaHidden[k] += deltaHidden[j] * neuron.weights[k] * 
                                    hiddenLayers.get(l - 1).neurons[k].output * (1 - hiddenLayers.get(l - 1).neurons[k].output);
                }
                
            }
        }

        deltaHidden = newDeltaHidden; // Update delta for next iteration (previous layer)
    }
    }

    /*
    public double[] feedForward(double[] input) {
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activate(input);
        }

        double output = outputNeuron.activate(hiddenOutputs);
        return new double[] { output };

    }
        */

    public double lossFunction(double[] inp, double expected, double predictedOutput) {

        double e = 1e-7; // to gaurd against log(0)
        double loss = 0.0;

        loss = expected * Math.log(Math.max(e, predictedOutput))
            + (1 - expected) * Math.log(Math.max(e, 1 - predictedOutput));
        

        return -loss;


    }

    public Vector<Double> getLosses()
    {
        return this.losses;
    }
}
