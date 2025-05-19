package com;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.Evaluation;
import weka.core.Utils;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Attribute;
import java.io.File;
import java.util.Random;
import java.util.ArrayList;

/**
 * Enhanced Stock Predictor with feature engineering for cryptocurrency price data
 * Features: Technical indicators, moving averages, volatility measures
 */
public class CryptoStockPredictor {

    public void DT() {
        System.out.println("Enhanced Cryptocurrency Prediction with Feature Engineering");
        try {
            String trainingFile = "src/data/BTC_train.csv";
            String testingFile = "src/data/BTC_test.csv";
            
            System.out.println("Starting enhanced cryptocurrency prediction...");
            
            // Load and enhance data with technical indicators
            Instances trainingData = loadAndEnhanceData(trainingFile);
            Instances testingData = loadAndEnhanceData(testingFile);
            
            // Display basic dataset information
            System.out.println("\nEnhanced training data: " + trainingData.numInstances() + 
                              " instances with " + trainingData.numAttributes() + " attributes");
            
            // Convert the output (class) attribute to nominal with better class definition
            trainingData = convertOutputToNominalEnhanced(trainingData);
            testingData = convertOutputToNominalEnhanced(testingData);
            
            // Set the class index
            int classIndex = trainingData.numAttributes() - 1;
            trainingData.setClassIndex(classIndex);
            testingData.setClassIndex(classIndex);
            
            System.out.println("\nClass attribute: " + trainingData.classAttribute().name());
            analyzeClassDistribution(trainingData);
            
            // Try multiple algorithms and compare
            System.out.println("\n=== Comparing Multiple Algorithms ===");
            
            // 1. Enhanced J48 with better parameters
            J48 j48 = new J48();
            String[] j48Options = new String[10];
            j48Options[0] = "-C"; j48Options[1] = "0.1";   // Lower confidence = less pruning
            j48Options[2] = "-M"; j48Options[3] = "10";    // Higher minimum instances per leaf
            j48Options[4] = "-A";                          // Don't perform subtree raising
            j48Options[5] = "-B";                          // Use binary splits
            j48Options[6] = "-J";                          // Don't use MDL correction
            j48Options[7] = "-Q"; j48Options[8] = "1";     // Seed for randomization
            j48Options[9] = "-S";                          // Don't perform subtree simplification
            j48.setOptions(j48Options);
            
            evaluateModel(j48, trainingData, testingData, "Enhanced J48");
            
            // 2. Random Forest (often better for noisy data)
            RandomForest rf = new RandomForest();
            String[] rfOptions = new String[6];
            rfOptions[0] = "-I"; rfOptions[1] = "100";    // Number of trees
            rfOptions[2] = "-K"; rfOptions[3] = "0";      // Number of features (0 = sqrt)
            rfOptions[4] = "-S"; rfOptions[5] = "1";      // Random seed
            rf.setOptions(rfOptions);
            
            evaluateModel(rf, trainingData, testingData, "Random Forest");
            
            // 3. k-NN (good for pattern recognition in time series)
            IBk knn = new IBk();
            String[] knnOptions = new String[2];
            knnOptions[0] = "-K"; knnOptions[1] = "7";    // Number of neighbors
            knn.setOptions(knnOptions);
            
            evaluateModel(knn, trainingData, testingData, "k-NN (k=7)");
            
        } catch (Exception e) {
            System.err.println("Error in enhanced prediction process:");
            e.printStackTrace();
        }
    }
    
    /**
     * Load CSV data and add technical indicators
     */
    private static Instances loadAndEnhanceData(String csvFilePath) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvFilePath));
        Instances originalData = loader.getDataSet();
        
        // Create enhanced dataset with technical indicators
        return addTechnicalIndicators(originalData);
    }
    
    /**
     * Add technical indicators as new features
     */
    private static Instances addTechnicalIndicators(Instances data) throws Exception {
        // Create attribute info for new features
        ArrayList<Attribute> attributes = new ArrayList<>();
        
        // Add original attributes EXCEPT the last one (which is the target/output)
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            attributes.add(data.attribute(i));
        }
        
        // Add new technical indicator attributes (keep them as features, not targets)
        attributes.add(new Attribute("SMA_5"));      // Simple Moving Average 5
        attributes.add(new Attribute("SMA_10"));     // Simple Moving Average 10
        attributes.add(new Attribute("RSI"));        // Relative Strength Index
        attributes.add(new Attribute("Volatility")); // Price volatility
        attributes.add(new Attribute("Price_Change_Pct")); // Price change percentage
        attributes.add(new Attribute("High_Low_Ratio"));   // High/Low ratio
        attributes.add(new Attribute("Volume_Price_Trend")); // If volume data available
        
        // Add the original target/output column LAST
        attributes.add(data.attribute(data.numAttributes() - 1));
        
        // Create new dataset
        Instances enhancedData = new Instances("EnhancedCrypto", attributes, data.numInstances());
        
        // Calculate and add technical indicators for each instance
        for (int i = 0; i < data.numInstances(); i++) {
            Instance originalInstance = data.instance(i);
            double[] values = new double[attributes.size()];
            
            // Copy original values EXCEPT the last one (target column)
            for (int j = 0; j < data.numAttributes() - 1; j++) {
                values[j] = originalInstance.value(j);
            }
            
            double open = originalInstance.value(0);
            double high = originalInstance.value(1);
            double low = originalInstance.value(2);
            double close = originalInstance.value(3);
            
            // Calculate SMA_5 (if enough data points)
            values[data.numAttributes() - 1] = calculateSMA(data, i, 5, 3); // Close price index = 3
            
            // Calculate SMA_10
            values[data.numAttributes()] = calculateSMA(data, i, 10, 3);
            
            // Calculate RSI (simplified version)
            values[data.numAttributes() + 1] = calculateRSI(data, i, 14, 3);
            
            // Calculate volatility (high-low ratio)
            values[data.numAttributes() + 2] = (high - low) / close;
            
            // Price change percentage from open to close
            values[data.numAttributes() + 3] = ((close - open) / open) * 100;
            
            // High/Low ratio
            values[data.numAttributes() + 4] = high / low;
            
            // Volume-Price Trend (placeholder - set to 0 if no volume data)
            values[data.numAttributes() + 5] = 0.0; // Replace with actual volume calculation if available
            
            // Copy the original target value to the last position
            values[attributes.size() - 1] = originalInstance.value(data.numAttributes() - 1);
            
            // Create and add new instance
            Instance newInstance = new DenseInstance(1.0, values);
            enhancedData.add(newInstance);
        }
        
        return enhancedData;
    }
    
    /**
     * Calculate Simple Moving Average
     */
    private static double calculateSMA(Instances data, int currentIndex, int period, int priceIndex) {
        if (currentIndex < period - 1) {
            return data.instance(currentIndex).value(priceIndex); // Not enough data, return current price
        }
        
        double sum = 0.0;
        for (int i = currentIndex - period + 1; i <= currentIndex; i++) {
            sum += data.instance(i).value(priceIndex);
        }
        return sum / period;
    }
    
    /**
     * Calculate Relative Strength Index (simplified)
     */
    private static double calculateRSI(Instances data, int currentIndex, int period, int priceIndex) {
        if (currentIndex < period) {
            return 50.0; // Neutral RSI when not enough data
        }
        
        double gains = 0.0;
        double losses = 0.0;
        
        for (int i = currentIndex - period + 1; i <= currentIndex; i++) {
            if (i > 0) {
                double change = data.instance(i).value(priceIndex) - data.instance(i-1).value(priceIndex);
                if (change > 0) {
                    gains += change;
                } else {
                    losses += Math.abs(change);
                }
            }
        }
        
        if (losses == 0) return 100.0;
        
        double avgGain = gains / period;
        double avgLoss = losses / period;
        double rs = avgGain / avgLoss;
        return 100 - (100 / (1 + rs));
    }
    
    /**
     * Enhanced class conversion with better thresholds using modern Weka API
     */
    private static Instances convertOutputToNominalEnhanced(Instances data) throws Exception {
        // First, analyze the distribution of the output values to set better thresholds
        int outputIndex = data.numAttributes() - 1;
        double[] outputValues = new double[data.numInstances()];
        
        for (int i = 0; i < data.numInstances(); i++) {
            outputValues[i] = data.instance(i).value(outputIndex);
        }
        
        // DEBUG: Print some sample values to understand the data
        System.out.println("\n=== DEBUG: Output column analysis ===");
        System.out.println("Output column name: " + data.attribute(outputIndex).name());
        System.out.println("First 10 output values:");
        for (int i = 0; i < Math.min(10, data.numInstances()); i++) {
            System.out.println("  Instance " + i + ": " + outputValues[i]);
        }
        
        // Calculate percentiles for better class boundaries
        java.util.Arrays.sort(outputValues);
        int n = outputValues.length;
        double min = outputValues[0];
        double max = outputValues[n-1];
        double lowerThreshold = outputValues[n/3];     // 33rd percentile
        double upperThreshold = outputValues[2*n/3];   // 67th percentile
        
        System.out.println("Min value: " + min + ", Max value: " + max);
        System.out.println("Class thresholds - Lower: " + lowerThreshold + ", Upper: " + upperThreshold);
        System.out.println("======================================\n");
        
        // Create class values using ArrayList (modern Weka API)
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("DOWN");   // Below lower threshold
        classValues.add("STABLE"); // Between thresholds
        classValues.add("UP");     // Above upper threshold
        
        Attribute classAttr = new Attribute("PriceDirection", classValues);
        
        // Create new dataset with enhanced classes
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            attributes.add(data.attribute(i));
        }
        attributes.add(classAttr);
        
        Instances newData = new Instances("EnhancedClasses", attributes, data.numInstances());
        
        // Convert instances with new class values
        for (int i = 0; i < data.numInstances(); i++) {
            Instance originalInstance = data.instance(i);
            double[] values = new double[attributes.size()];
            
            // Copy all attribute values except the last one
            for (int j = 0; j < data.numAttributes() - 1; j++) {
                values[j] = originalInstance.value(j);
            }
            
            // Determine class based on thresholds
            double outputValue = originalInstance.value(outputIndex);
            if (outputValue <= lowerThreshold) {
                values[attributes.size() - 1] = 0; // DOWN
            } else if (outputValue >= upperThreshold) {
                values[attributes.size() - 1] = 2; // UP
            } else {
                values[attributes.size() - 1] = 1; // STABLE
            }
            
            Instance newInstance = new DenseInstance(1.0, values);
            newData.add(newInstance);
        }
        
        return newData;
    }
    
    /**
     * Evaluate a model and print results
     */
    private static void evaluateModel(weka.classifiers.Classifier classifier, 
                                    Instances trainingData, Instances testingData, 
                                    String modelName) throws Exception {
        System.out.println("\n=== " + modelName + " ===");
        
        // Build classifier
        classifier.buildClassifier(trainingData);
        
        // Cross-validation on training data
        Evaluation evalTrain = new Evaluation(trainingData);
        evalTrain.crossValidateModel(classifier, trainingData, 10, new Random(1));
        System.out.println("Cross-validation accuracy: " + String.format("%.2f%%", evalTrain.pctCorrect()));
        
        // Test set evaluation
        Evaluation evalTest = new Evaluation(trainingData);
        evalTest.evaluateModel(classifier, testingData);
        System.out.println("Test set accuracy: " + String.format("%.2f%%", evalTest.pctCorrect()));
        System.out.println("Test set F1 Score: " + String.format("%.4f", evalTest.weightedFMeasure()));
        
        // Show confusion matrix for test set
        System.out.println("Confusion Matrix:");
        System.out.println(evalTest.toMatrixString());
    }
    
    /**
     * Analyze class distribution in the dataset
     */
    private static void analyzeClassDistribution(Instances data) {
        int[] classCounts = new int[data.numClasses()];
        
        for (int i = 0; i < data.numInstances(); i++) {
            int classValue = (int) data.instance(i).classValue();
            classCounts[classValue]++;
        }
        
        System.out.println("\nClass distribution in training data:");
        for (int i = 0; i < classCounts.length; i++) {
            System.out.println("  Class '" + data.classAttribute().value(i) + 
                              "': " + classCounts[i] + " instances (" + 
                              String.format("%.2f", (double)classCounts[i]/data.numInstances()*100) + "%)");
        }
    }
}