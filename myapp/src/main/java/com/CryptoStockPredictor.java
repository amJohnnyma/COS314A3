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
    // Default J48 parameters
    private double confidenceFactor = 0.1;    // Lower confidence = less pruning
    private int minNumObj = 10;               // Higher minimum instances per leaf
    private boolean unpruned = false;         // Do pruning
    private boolean useBinarySplits = true;   // Use binary splits
    private boolean noSubtreeRaising = true;  // Don't perform subtree raising
    private boolean noCleanup = false;        // Do cleanup after build
    private boolean collapseTree = true;      // Do collapse the tree
    private boolean useMDLcorrection = false; // Don't use MDL correction
    private int seed = 1;                     // Seed for randomization
    private boolean subtreeRaising = false;   // Don't perform subtree raising
    
    // Model evaluation metrics
    private double trainingAccuracy = 0.0;
    private double cvAccuracy = 0.0;
    private double testAccuracy = 0.0;
    private double testF1Score = 0.0;
    private int treeSize = 0;
    private int numLeaves = 0;

    // Getters for model metrics
    public double getTrainingAccuracy() { return trainingAccuracy; }
    public double getCVAccuracy() { return cvAccuracy; }
    public double getTestAccuracy() { return testAccuracy; }
    public double getTestF1Score() { return testF1Score; }
    public int getTreeSize() { return treeSize; }
    public int getNumLeaves() { return numLeaves; }
    
    // Setters and getters for J48 parameters
    public double getConfidenceFactor() { return confidenceFactor; }
    public void setConfidenceFactor(double confidenceFactor) { this.confidenceFactor = confidenceFactor; }
    
    public int getMinNumObj() { return minNumObj; }
    public void setMinNumObj(int minNumObj) { this.minNumObj = minNumObj; }
    
    public boolean isUnpruned() { return unpruned; }
    public void setUnpruned(boolean unpruned) { 
        this.unpruned = unpruned; 
        // If setting to unpruned, we need to ensure subtree options are compatible
        if (unpruned) {
            // When unpruned is true, subtree raising options should be ignored
            // but we'll set them to their default values to avoid confusion
            this.noSubtreeRaising = false;
            this.subtreeRaising = false;
        }
    }
    
    public boolean isUseBinarySplits() { return useBinarySplits; }
    public void setUseBinarySplits(boolean useBinarySplits) { this.useBinarySplits = useBinarySplits; }
    
    public boolean isNoSubtreeRaising() { return noSubtreeRaising; }
    public void setNoSubtreeRaising(boolean noSubtreeRaising) { this.noSubtreeRaising = noSubtreeRaising; }
    
    public boolean isNoCleanup() { return noCleanup; }
    public void setNoCleanup(boolean noCleanup) { this.noCleanup = noCleanup; }
    
    public boolean isCollapseTree() { return collapseTree; }
    public void setCollapseTree(boolean collapseTree) { this.collapseTree = collapseTree; }
    
    public boolean isUseMDLcorrection() { return useMDLcorrection; }
    public void setUseMDLcorrection(boolean useMDLcorrection) { this.useMDLcorrection = useMDLcorrection; }
    
    public int getSeed() { return seed; }
    public void setSeed(int seed) { this.seed = seed; }
    
    public boolean isSubtreeRaising() { return subtreeRaising; }
    public void setSubtreeRaising(boolean subtreeRaising) { this.subtreeRaising = subtreeRaising; }
    
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
            
            // Build J48 with current parameter settings
            J48 j48 = buildJ48Classifier();
            
            evaluateModel(j48, trainingData, testingData, "Enhanced J48");
            
        } catch (Exception e) {
            System.err.println("Error in enhanced prediction process:");
            e.printStackTrace();
        }
    }
    
    /**
     * Build J48 classifier with current parameter values
     */
    private J48 buildJ48Classifier() throws Exception {
        J48 j48 = new J48();
        ArrayList<String> options = new ArrayList<>();
        
        if (unpruned)
        {
            options.add("-C"); options.add(String.valueOf(confidenceFactor));
        }
        options.add("-M"); options.add(String.valueOf(minNumObj));
        
        if (unpruned) {
            options.add("-U");
            // Skip subtree raising options when unpruned is true
            // as they're incompatible according to Weka's implementation
        } else {
            // Only add subtree options when we're not using unpruned trees
            if (noSubtreeRaising) options.add("-S");
            if (subtreeRaising) options.add("-A");
        }
        
        if (useBinarySplits) options.add("-B");
        if (noCleanup) options.add("-L");
        if (!collapseTree) options.add("-O");
        if (!useMDLcorrection) options.add("-J");
        
        options.add("-Q"); options.add(String.valueOf(seed));
        
        String[] optionsArray = options.toArray(new String[0]);
        j48.setOptions(optionsArray);
        
        return j48;
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
    private void evaluateModel(weka.classifiers.Classifier classifier, 
                                    Instances trainingData, Instances testingData, 
                                    String modelName) throws Exception {
        System.out.println("\n=== " + modelName + " ===");
        
        // Build classifier
        classifier.buildClassifier(trainingData);
        
        // Get tree information if it's a J48
        if (classifier instanceof J48) {
            J48 j48 = (J48) classifier;
            treeSize = (int)j48.measureTreeSize();
            numLeaves = (int)j48.measureNumLeaves();
            System.out.println("Tree size (number of nodes): " + treeSize);
            System.out.println("Number of leaves: " + numLeaves);
            System.out.println("Tree structure:");
            System.out.println(j48.toString());
        }
        
        // Training set accuracy (new addition)
        Evaluation evalOnTrain = new Evaluation(trainingData);
        evalOnTrain.evaluateModel(classifier, trainingData);
        trainingAccuracy = evalOnTrain.pctCorrect();
        System.out.println("Training set accuracy: " + String.format("%.2f%%", trainingAccuracy));
        
        // Cross-validation on training data
        Evaluation evalTrain = new Evaluation(trainingData);
        evalTrain.crossValidateModel(classifier, trainingData, 10, new Random(1));
        cvAccuracy = evalTrain.pctCorrect();
        System.out.println("Cross-validation accuracy: " + String.format("%.2f%%", cvAccuracy));
        
        // Test set evaluation
        Evaluation evalTest = new Evaluation(trainingData);
        evalTest.evaluateModel(classifier, testingData);
        testAccuracy = evalTest.pctCorrect();
        testF1Score = evalTest.weightedFMeasure();
        System.out.println("Test set accuracy: " + String.format("%.2f%%", testAccuracy));
        System.out.println("Test set F1 Score: " + String.format("%.4f", testF1Score));
        
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