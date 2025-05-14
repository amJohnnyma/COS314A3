package com;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import weka.core.pmml.Array;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.*;

public class MLP implements Serializable{

    private List<double[]> inputsVal = new ArrayList<>(); /// full input of each
    private List<double[]> inputs = new ArrayList<>(); // 5 inputs per // one input per feature (Open, high, low, close,
    private List<Layer> hiddenLayers = new ArrayList<>();
    private List<String> inputsTime = new ArrayList<>(); // Orderof each 5 inputs//might be useless
    private List<Double> labels = new ArrayList<>(); // expected output (last input)
    private Neuron outputNeuron; // one per label
    private int numSamples = 0;
    private double learningRate = 0.0;

    // Data collection
    public Vector<Double> losses = new Vector<Double>();
    public Vector<Double> avgGradient = new Vector<Double>();
    private Vector<Double> expectedValues = new Vector<Double>();
    private Vector<Double> predictedValues = new Vector<Double>();
    private Vector<Double> accuracyOverTime = new Vector<Double>();
    private Vector<Double> recallOverTime = new Vector<Double>();
    private Vector<Double> f1ScoreOverTime = new Vector<Double>();
    public Vector<Double> avgWeights = new Vector<>();
    public Vector<Double> avgBiases = new Vector<>();
    Vector<Long> epochTimes = new Vector<>();

    ////
    public MLP()
    {

    }
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
        for (int i = 0; i < hiddenL; i++) {
            Layer layer = new Layer(currentInputSize, hiddenSize, seed);
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

    public void trainNetwork(int iterations, int batchSize) {// https://chatgpt.com/share/6824badd-a01c-8012-bb91-21d7c211a6a0

   
        for (int k = 0; k < iterations; k++) // epochs
        {
            long start = System.nanoTime();
            shuffle(inputs, labels);
            for (int i = 0; i < inputs.size(); i += batchSize) {
                // bacth implementation
                List<double[]> batchInputs = new ArrayList<>();
                List<Double> batchLabels = new ArrayList<>();

                for (int j = i; j < i + batchSize && j < inputs.size(); j++) {
                    batchInputs.add(inputs.get(j));
                    batchLabels.add(labels.get(j));
                }

                double batchLoss = 0.0;

                List<double[]> activations = new ArrayList<>();
                List<Double> predictions = new ArrayList<>();

                for (int b = 0; b < batchInputs.size(); b++) {
                    double[] input = batchInputs.get(b);
                    double[] out = input;

                    for (Layer layer : hiddenLayers) {
                        out = layer.forward(out);
                    }

                    double prediction = outputNeuron.activate(out);
                    activations.add(out);
                    predictions.add(prediction);

                    double expected = batchLabels.get(b);
                    batchLoss += lossFunction(out, expected, prediction); // accumulate loss
                }

                batchLoss /= batchInputs.size();
                losses.add(batchLoss);

                // Backward pass on each sample
                /*
                for (int b = 0; b < batchInputs.size(); b++) {
                    backward(activations.get(b), batchLabels.get(b), predictions.get(b));
                }
                    */
                backwardBatch(batchInputs, batchLabels, predictions);
                // end of batch implementation
                long end = System.nanoTime();

                epochTimes.add((end-start)/ 1_000_000); // ms
                logWeightsAndBiases();

                /*
                 * double[] input = inputs.get(i);
                 * double[] out = input;
                 * double expectedOutput = labels.get(i);
                 * for (Layer layer : hiddenLayers) {
                 * out = layer.forward(out); // feedforward each neuron in the layer
                 * }
                 * double prediction = outputNeuron.activate(out); // final prediction
                 * 
                 * double lf = lossFunction(out, expectedOutput, prediction);
                 * losses.add(lf);
                 * 
                 * // System.out.println("Expected: " + expectedOutput + " Predicted: " +
                 * // prediction);
                 * // System.out.println("Loss: " + lf);
                 * 
                 * backward(out, expectedOutput, prediction);
                 */

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

    public void backward(double[] input, double expectedOutput, double prediction) {
        // Step 1: Compute the error (delta) at the output layer
        double delta = prediction - expectedOutput;

        double agrad = 0.0;
        // Update output neuron weights (outputNeuron is the final layer neuron)
        for (int i = 0; i < hiddenLayers.get(hiddenLayers.size() - 1).neurons.length; i++) {
            double gradient = delta * hiddenLayers.get(hiddenLayers.size() - 1).neurons[i].output;
            outputNeuron.updateWeights(learningRate, gradient);
            agrad += gradient;
        }
        avgGradient.add(agrad);

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

                double reluDeriv = (neuron.z > 0) ? 1.0 : 0.0; // ReLU derivative

                // Delta for this neuron (local delta for hidden neurons)
                double deltalocal = 0.0;

                // If it's the last hidden layer (before output layer), we use the delta
                // propagated from the output layer
                if (l == hiddenLayers.size() - 1) {
                    deltalocal = delta * reluDeriv; // Local delta from the output layer
                } else {
                    // For other hidden layers, use the error propagated from the next layer
                    for (int k = 0; k < hiddenLayers.get(l + 1).neurons.length; k++) {
                        // Backpropagate the delta from the next layer
                        deltalocal += hiddenLayers.get(l + 1).neurons[k].weights[j] * deltaHidden[k];
                    }
                    deltalocal *= reluDeriv; // Apply the derivative of the activation function (ReLU)
                }

                // for data
                double avgGrad = 0.0;
                // Backpropagate delta through each neuron
                for (int k = 0; k < neuron.weights.length; k++) {
                    double inputToThisNeuron = neuron.input[k]; // saved during forward pass
                    double gradient = deltalocal * inputToThisNeuron;
                    neuron.weights[k] -= learningRate * gradient;
                    // System.out.println("Gradient: " + gradient);
                    avgGrad += gradient;

                }
                avgGradient.add(avgGrad);

                // Update the neuron bias
                neuron.bias -= learningRate * deltalocal;

                newDeltaHidden[j] = deltalocal;
            }

            deltaHidden = newDeltaHidden; // Update delta for next iteration (previous layer)
        }
    }

    public void backwardBatch(List<double[]> batchInputs, List<Double> batchLabels, List<Double> batchPredictions) {
        int batchSize = batchInputs.size();

        // Initialize accumulators for output neuron
        double[] outputWeightGrads = new double[outputNeuron.weights.length];
        double outputBiasGrad = 0;

        // Initialize accumulators for hidden layers
        List<double[][]> hiddenWeightGrads = new ArrayList<>();
        List<double[]> hiddenBiasGrads = new ArrayList<>();

        for (Layer layer : hiddenLayers) {
            hiddenWeightGrads.add(new double[layer.neurons.length][layer.neurons[0].weights.length]);
            hiddenBiasGrads.add(new double[layer.neurons.length]);
        }

        // === Accumulate gradients over all samples ===
        for (int b = 0; b < batchSize; b++) {
            double[] input = batchInputs.get(b);
            double expectedOutput = batchLabels.get(b);
            double prediction = batchPredictions.get(b);

            double delta = prediction - expectedOutput;

            // Output neuron gradient
            for (int i = 0; i < hiddenLayers.get(hiddenLayers.size() - 1).neurons.length; i++) {
                double hiddenOut = hiddenLayers.get(hiddenLayers.size() - 1).neurons[i].output;
                outputWeightGrads[i] += delta * hiddenOut;
            }
            outputBiasGrad += delta;

            // Hidden layers backward
            double[] deltaHidden = new double[hiddenLayers.get(hiddenLayers.size() - 1).neurons.length];

            for (int l = hiddenLayers.size() - 1; l >= 0; l--) {
                Layer layer = hiddenLayers.get(l);
                double[] newDeltaHidden = new double[layer.neurons.length];

                for (int j = 0; j < layer.neurons.length; j++) {
                    Neuron neuron = layer.neurons[j];
                    double reluDeriv = (neuron.z > 0) ? 1.0 : 0.0;
                    double deltalocal = 0.0;

                    if (l == hiddenLayers.size() - 1) {
                        deltalocal = delta * reluDeriv;
                    } else {
                        for (int k = 0; k < hiddenLayers.get(l + 1).neurons.length; k++) {
                            deltalocal += hiddenLayers.get(l + 1).neurons[k].weights[j] * deltaHidden[k];
                        }
                        deltalocal *= reluDeriv;
                    }

                    // Accumulate gradients
                    for (int k = 0; k < neuron.weights.length; k++) {
                        hiddenWeightGrads.get(l)[j][k] += deltalocal * neuron.input[k];
                    }
                    hiddenBiasGrads.get(l)[j] += deltalocal;

                    newDeltaHidden[j] = deltalocal;
                }

                deltaHidden = newDeltaHidden;
            }
        }

        // === Apply averaged gradients ===

        // Output neuron update
        for (int i = 0; i < outputNeuron.weights.length; i++) {
            outputNeuron.weights[i] -= learningRate * (outputWeightGrads[i] / batchSize);
        }
        outputNeuron.bias -= learningRate * (outputBiasGrad / batchSize);

        // Hidden layers update
        for (int l = 0; l < hiddenLayers.size(); l++) {
            Layer layer = hiddenLayers.get(l);
            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                for (int k = 0; k < neuron.weights.length; k++) {
                    neuron.weights[k] -= learningRate * (hiddenWeightGrads.get(l)[j][k] / batchSize);
                }
                neuron.bias -= learningRate * (hiddenBiasGrads.get(l)[j] / batchSize);
            }
        }
    }

    public double lossFunction(double[] inp, double expected, double predictedOutput) {

        double e = 1e-7; // to gaurd against log(0)
        double loss = 0.0;

        loss = expected * Math.log(Math.max(e, predictedOutput))
                + (1 - expected) * Math.log(Math.max(e, 1 - predictedOutput));

        return -loss;

    }

    public Vector<Double> getLosses() {
        return this.losses;
    }

    public Vector<Double> getGradients() {
        return this.avgGradient;
    }

    public static void shuffle(List<double[]> inputs, List<Double> labels) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            indices.add(i);
        }

        Collections.shuffle(indices); // Randomly shuffle indices

        List<double[]> shuffledInputs = new ArrayList<>();
        List<Double> shuffledLabels = new ArrayList<>();

        for (int index : indices) {
            shuffledInputs.add(inputs.get(index));
            shuffledLabels.add(labels.get(index));
        }

        // Replace original lists
        inputs.clear();
        inputs.addAll(shuffledInputs);

        labels.clear();
        labels.addAll(shuffledLabels);
    }

        // Method to save the model to a file
    public void saveModel(String filename) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {
            out.writeObject(this); // Write the entire MLP object to the file
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Method to load the model from a file
    public static MLP loadModel(String filename) {
        MLP mlp = null;
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename))) {
            mlp = (MLP) in.readObject(); // Read the object from the file
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return mlp;
    }

    private void logWeightsAndBiases() {
    double totalWeight = 0.0;
    int weightCount = 0;
    double totalBias = 0.0;
    int biasCount = 0;

    for (Layer layer : hiddenLayers) {
        for (Neuron neuron : layer.neurons) {
            for (double w : neuron.weights) {
                totalWeight += w;
                weightCount++;
            }
            totalBias += neuron.bias;
            biasCount++;
        }
    }

    // Output neuron
    for (double w : outputNeuron.weights) {
        totalWeight += w;
        weightCount++;
    }
    totalBias += outputNeuron.bias;
    biasCount++;

    avgWeights.add(totalWeight / weightCount);
    avgBiases.add(totalBias / biasCount);
}

/*public void saveModel(String filename) {
    try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
        writer.write(inputSize + "," + hiddenSize + "," + outputSize);
        writer.newLine();

        for (double[] layerWeights : weights) {
            for (double weight : layerWeights) {
                writer.write(weight + ",");
            }
            writer.newLine();
        }

        for (double[] layerBiases : biases) {
            for (double bias : layerBiases) {
                writer.write(bias + ",");
            }
            writer.newLine();
        }

    } catch (IOException e) {
        e.printStackTrace();
    }
}
 public void loadModel(String filename) {
    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
        String[] sizes = reader.readLine().split(",");
        inputSize = Integer.parseInt(sizes[0]);
        hiddenSize = Integer.parseInt(sizes[1]);
        outputSize = Integer.parseInt(sizes[2]);

        weights = new ArrayList<>();
        biases = new ArrayList<>();

        for (int i = 0; i < numLayers - 1; i++) {
            String[] weightLine = reader.readLine().split(",");
            double[] layerWeights = Arrays.stream(weightLine)
                                          .filter(s -> !s.isEmpty())
                                          .mapToDouble(Double::parseDouble)
                                          .toArray();
            weights.add(layerWeights);
        }

        for (int i = 0; i < numLayers - 1; i++) {
            String[] biasLine = reader.readLine().split(",");
            double[] layerBiases = Arrays.stream(biasLine)
                                         .filter(s -> !s.isEmpty())
                                         .mapToDouble(Double::parseDouble)
                                         .toArray();
            biases.add(layerBiases);
        }

    } catch (IOException e) {
        e.printStackTrace();
    }
}
*/
}
