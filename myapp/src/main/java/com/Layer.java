package com;

import java.io.Serializable;

class Layer implements Serializable{
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