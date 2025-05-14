package com;
import java.util.*;

class Neuron
{

    double[] weights; //weights to each othe
    double bias;
    double output;
    double delta;

    public Neuron(int inputSize, long seed) {
        weights = new double[inputSize];
        Random r = new Random(seed); //use seed somehow
        for (int i = 0; i < inputSize; i++) { //small random weight for each input
            weights[i] = r.nextGaussian() * 0.01; // small random init
        }
        bias = 0;

    }

    public double activate(double[] inputs) {
        double z = bias + weightedSum(inputs);
        output = sigmoid(z); 
        return output;

        /*
            Sigmoid: σ(z)=11+e−zσ(z)=1+e−z1​
            ReLU (Rectified Linear Unit): f(z)=max⁡(0,z)f(z)=max(0,z)
            Tanh (Hyperbolic Tangent): tanh⁡(z)=21+e−2z–1tanh(z)=1+e−2z2​–1
         */
    }

    public double weightedSum(double[] inputs)
    {
        double sum = bias;
        for(int k =0; k < inputs.length;k++)
        {
            sum += weights[k] * inputs[k];
        }
        return sum;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double gradientLoss(double deltaOutput, double input) {
        // Compute the gradient of the loss with respect to the weight
        // deltaOutput: the error term at this neuron (how much the output of the neuron deviates from the expected output)
        // input: the input to this neuron (the feature value or output from the previous layer)
        
        // This assumes the sigmoid activation function, so we need to multiply by the derivative of the sigmoid
        // d(sigmoid)/dz = sigmoid(z) * (1 - sigmoid(z))
        
        double gradient = deltaOutput * input; // Gradient with respect to the weight
        return gradient;
    }

    public void updateWeights(double learningRate, double deltaWeight)
    {
         
        for(int i =0; i < weights.length; i++)
        {
            weights[i] -= (learningRate * deltaWeight);
        }
    }
}