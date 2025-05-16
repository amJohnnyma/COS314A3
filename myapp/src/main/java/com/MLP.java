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

public class MLP implements Serializable {

    private List<double[]> inputsVal = new ArrayList<>(); /// full input of each
    private List<double[]> inputs = new ArrayList<>(); // 5 inputs per // one input per feature (Open, high, low, close,
    private List<Layer> hiddenLayers = new ArrayList<>();
    private List<String> inputsTime = new ArrayList<>(); // Orderof each 5 inputs//might be useless
    private List<Double> labels = new ArrayList<>(); // expected output (last input)
    private Neuron outputNeuron; // one per label
    private int numSamples = 0;
    private double learningRate = 0.0;
    private long seed = 0;

    // Data collection
    public Vector<Double> losses = new Vector<Double>();
    public Vector<Double> avgGradient = new Vector<Double>();
    public Vector<Double> deltaValues = new Vector<Double>();
    private Vector<Double> predictedValues = new Vector<Double>();
    private Vector<Double> accuracyOverTime = new Vector<Double>();
    private Vector<Double> recallOverTime = new Vector<Double>();
    private Vector<Double> f1ScoreOverTime = new Vector<Double>();
    public Vector<Double> avgWeights = new Vector<>();
    public Vector<Double> avgBiases = new Vector<>();
    Vector<Long> epochTimes = new Vector<>();    

    ////
    public MLP() {

    }

    public MLP(String file, int hiddenSize, int hiddenL, int inputSize, long seed, double learningRate) // constructor
    {
        this.seed = seed;
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
        double[] min = new double[inputSize];
        double[] max = new double[inputSize];

        // Initialize min/max
        Arrays.fill(min, Double.POSITIVE_INFINITY);
        Arrays.fill(max, Double.NEGATIVE_INFINITY);

        // 1. Find min and max per column
        for (double[] row : inputsVal) {
            for (int i = 0; i < inputSize; i++) {
                if (row[i] < min[i]) min[i] = row[i];
                if (row[i] > max[i]) max[i] = row[i];
            }
        }

        // 2. Apply min-max normalization to scale to [-1, 1]
        for (double[] row : inputsVal) {
            for (int i = 0; i < inputSize; i++) {
                double range = max[i] - min[i];
                if (range != 0) {
                    row[i] = 2 * (row[i] - min[i]) / range - 1;  // scaled to [-1, 1]
                } else {
                    row[i] = 0; // If constant feature, set to 0
                }
            }
        }

        for (double[] row : inputsVal) {
            double[] input = Arrays.copyOfRange(row, 0, 5); // Already normalized
            double label = row[5]; // Output label stays unchanged
            inputs.add(input);
            labels.add(label);

            //debug

        }

            // for(int i = 0; i < inputs.size(); i++)
            // {
            //     for(int k = 0; k < inputs.get(i).length; k++)
            //     {
            //       //  if(inputs.get(i)[k] < -1)
            //         System.out.println(inputs.get(i)[k]);

            //       //  if(inputs.get(i)[k] > 1)
            //       //  System.out.println(inputs.get(i)[k]);

            //     }
            // }


     //   inputs.add(new double[] { -0.9, -0.8, -0.85, -0.82, -0.9 });
     //   labels.add(1.0);
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

    public void trainNetwork(int iterations, int batchSize, int patience, double minImpro) {// https://chatgpt.com/share/6824badd-a01c-8012-bb91-21d7c211a6a0

        double lastLoss = Double.MAX_VALUE;
        int epochsWithoutImprovement = 0;
        for (int k = 0; k < iterations; k++) // epochs
        {
            double epochLoss = 0.0;
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
                    double prediction = feedForward(out);
                   // activations.add(out);
                    predictions.add(prediction);

                    double expected = batchLabels.get(b);
                   batchLoss += lossFunction(out, expected, prediction); // accumulate loss
                }

                batchLoss /= batchInputs.size();
                losses.add(batchLoss);
                epochLoss += batchLoss;

                backwardBatch(batchInputs, batchLabels, predictions);
                // end of batch implementation

                // Update last loss for the next epoch comparison
                long end = System.nanoTime();

                epochTimes.add((end - start) / 1_000_000); // ms
                logWeightsAndBiases();

            }
            int numBatches = (inputs.size() + batchSize - 1) / batchSize;
            epochLoss /= numBatches;

            // Check if the improvement is below the threshold
            if (Math.abs(lastLoss - epochLoss) < minImpro) {
                epochsWithoutImprovement++;
            } else {
                epochsWithoutImprovement = 0; // Reset counter if improvement happened
            }

            // If the loss has not improved for 'patience' epochs, stop training
            if (epochsWithoutImprovement >= patience) {
                System.out.println("Early stopping triggered at epoch " + k);
                break; // Early stop
            }

            lastLoss = epochLoss;
        }

        


    }
public double testNetwork() {
    int correct = 0;
    double totalLoss = 0.0;

    for (int i = 0; i < inputs.size(); i++) {
        double[] input = inputs.get(i);
        double expected = labels.get(i);
        double prediction = feedForward(input);

        // Optional: Round prediction to 0 or 1
        int predRounded = prediction >= 0.5 ? 1 : 0;
        int labelRounded = expected >= 0.5 ? 1 : 0;

        if (predRounded == labelRounded) {
            correct++;
        }

        totalLoss += lossFunction(input, expected, prediction);
    }

    double accuracy = (double) correct / inputs.size();
    double avgLoss = totalLoss / inputs.size();

    if(accuracy >= 0.95)
    {
    System.out.println("Seed: " + seed);
    System.out.println("Average Loss: " + avgLoss);
    }

    return accuracy;
}

    //////////// Helper functions// Could probably be its own class
    public static boolean containsNumbers(String input) {
        Pattern pattern = Pattern.compile("\\d");
        Matcher matcher = pattern.matcher(input);
        return matcher.find();
    }

    public double feedForward(double[] out) {
    for (Layer layer : hiddenLayers) {
        out = layer.forward(out);
    }
    return outputNeuron.activate(out);
}


//Maybe refactor to use matrices
    public void backwardBatch(List<double[]> batchInputs, List<Double> batchLabels, List<Double> batchPredictions) {
        int batchSize = batchInputs.size();

        // Initialize accumulators for output neuron
        double[] outputWeightGrads = new double[outputNeuron.weights.length];
        double outputBiasGrad = 0;

        // Initialize accumulators for hidden layers
        List<double[][]> hiddenWeightGrads = new ArrayList<>();
        List<double[]> hiddenBiasGrads = new ArrayList<>();
        double deltaAvg = 0.0;

        for (Layer layer : hiddenLayers) {
            hiddenWeightGrads.add(new double[layer.neurons.length][layer.neurons[0].weights.length]);
            hiddenBiasGrads.add(new double[layer.neurons.length]);
        }

        //  Accumulate gradients over all samples 
        for (int b = 0; b < batchSize; b++) {
            double[] input = batchInputs.get(b);
            double expectedOutput = batchLabels.get(b);
            double prediction = batchPredictions.get(b);

            feedForward(input);

            double delta = (prediction - expectedOutput);
            deltaAvg+=delta;

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
                    // double reluDeriv = neuron.reluDerivative(neuron.z);
                    double sigmoidDeriv = neuron.output * (1.0 - neuron.output);

                    double deltalocal = 0.0;
                if (l == hiddenLayers.size() - 1) {
                    // Last hidden layer connects to output
                    deltalocal = delta * sigmoidDeriv;
                } else {
                    for (int k = 0; k < hiddenLayers.get(l + 1).neurons.length; k++) {
                        deltalocal += deltaHidden[k] * hiddenLayers.get(l + 1).neurons[k].weights[j];
                    }
                    deltalocal *= sigmoidDeriv;
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
        deltaAvg /= batchSize;
         deltaValues.add(deltaAvg);
       // outputBiasGrad /= batchSize;


        // Output neuron update
        for (int i = 0; i < outputNeuron.weights.length; i++) {
            outputNeuron.weights[i] -= learningRate * (outputWeightGrads[i] );
        }
        outputNeuron.bias -= learningRate * (outputBiasGrad );

        // Hidden layers update
        for (int l = 0; l < hiddenLayers.size(); l++) {
            Layer layer = hiddenLayers.get(l);
            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                for (int k = 0; k < neuron.weights.length; k++) {
                    neuron.weights[k] -= learningRate * (hiddenWeightGrads.get(l)[j][k] );
                    // neuron.updateWeights(learningRate, hiddenWeightGrads.get(l)[j][k] /
                    // batchSize);
                }
                neuron.bias -= learningRate * (hiddenBiasGrads.get(l)[j] );
            }
        }
    }

    public double lossFunction(double[] inp, double expected, double predictedOutput) {

        double e = 1e-9; // to gaurd against log(0)
        double loss = 0.0;

        loss = expected * Math.log(e+ predictedOutput)
                + (1 - expected) * Math.log(e + 1 - predictedOutput);

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
    /* From chat:
     ✅ What is safe?

    Adding new methods (like utility functions or training logic).

    Adding transient fields (they are ignored during serialization).

    Adding new non-transient fields with default values (but see warnings below).

⚠️ What can break deserialization?

If you do any of the following, deserialization of previously saved objects can fail with InvalidClassException:

    Change the class structure significantly, like:

        Renaming or removing fields.

        Changing field types.

    Change the class name or package name.

    Don’t manage serialVersionUID correctly (see below).


    Change Type	Safe?	Notes
Add method	✅	Fully safe.
Add field	⚠️	Safe if you use default values and set serialVersionUID.
Remove/rename field	❌	Breaks compatibility.
Change class/package name	❌	Breaks compatibility.
Add serialVersionUID	✅	Strongly recommended.
     */
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

    private double getCurrentLoss() {
        return losses.lastElement();
    }

}

/*
 * public void backward(double[] input, double expectedOutput, double
 * prediction) {
 * // Step 1: Compute the error (delta) at the output layer
 * double delta = prediction - expectedOutput;
 * 
 * double agrad = 0.0;
 * // Update output neuron weights (outputNeuron is the final layer neuron)
 * for (int i = 0; i < hiddenLayers.get(hiddenLayers.size() - 1).neurons.length;
 * i++) {
 * double gradient = delta * hiddenLayers.get(hiddenLayers.size() -
 * 1).neurons[i].output;
 * outputNeuron.updateWeights(learningRate, gradient);
 * agrad += gradient;
 * }
 * avgGradient.add(agrad);
 * 
 * // Update output neuron bias
 * outputNeuron.bias -= learningRate * delta;
 * 
 * // Step 2: Backpropagate the error through hidden layers
 * double[] deltaHidden = new double[hiddenLayers.get(hiddenLayers.size() -
 * 1).neurons.length];
 * 
 * // Loop through hidden layers backwards
 * for (int l = hiddenLayers.size() - 1; l >= 0; l--) {
 * Layer layer = hiddenLayers.get(l);
 * double[] newDeltaHidden = new double[layer.neurons.length];
 * 
 * // Calculate delta for each neuron in this layer
 * for (int j = 0; j < layer.neurons.length; j++) {
 * Neuron neuron = layer.neurons[j];
 * 
 * // double reluDeriv = (neuron.z > 0) ? 1.0 : 0.0; // ReLU derivative
 * double sigmoidDeriv = neuron.output * (1.0 - neuron.output);
 * 
 * // Delta for this neuron (local delta for hidden neurons)
 * double deltalocal = 0.0;
 * 
 * // If it's the last hidden layer (before output layer), we use the delta
 * // propagated from the output layer
 * if (l == hiddenLayers.size() - 1) {
 * deltalocal = delta * sigmoidDeriv; // Local delta from the output layer
 * } else {
 * // For other hidden layers, use the error propagated from the next layer
 * for (int k = 0; k < hiddenLayers.get(l + 1).neurons.length; k++) {
 * // Backpropagate the delta from the next layer
 * deltalocal += hiddenLayers.get(l + 1).neurons[k].weights[j] * deltaHidden[k];
 * }
 * deltalocal *= sigmoidDeriv; // Apply the derivative of the activation
 * function (ReLU)
 * }
 * 
 * // for data
 * double avgGrad = 0.0;
 * // Backpropagate delta through each neuron
 * for (int k = 0; k < neuron.weights.length; k++) {
 * double inputToThisNeuron = neuron.input[k]; // saved during forward pass
 * double gradient = deltalocal * inputToThisNeuron;
 * neuron.weights[k] -= learningRate * gradient;
 * // System.out.println("Gradient: " + gradient);
 * avgGrad += gradient;
 * 
 * }
 * avgGradient.add(avgGrad);
 * 
 * // Update the neuron bias
 * neuron.bias -= learningRate * deltalocal;
 * 
 * newDeltaHidden[j] = deltalocal;
 * }
 * 
 * deltaHidden = newDeltaHidden; // Update delta for next iteration (previous
 * layer)
 * }
 * }
 */