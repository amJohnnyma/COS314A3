package main.java.com;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.Evaluation;
import weka.core.Utils;
import java.io.File;
import java.io.FileReader;
import java.util.Random;

/**
 * Stock Predictor using J48 Decision Trees in Weka
 * Specifically designed for cryptocurrency price data with the format:
 * Open, High, Low, Close, Adj Close, Output
 */
public class CryptoStockPredictor {

    public void DT() {
        System.out.println("DT");
        try {
            String trainingFile = "src/data/BTC_train.csv";
            String testingFile = "src/data/BTC_test.csv";
            
            System.out.println("Starting cryptocurrency stock prediction with J48...");
            
            // Load data
            Instances trainingData = loadCSVData(trainingFile);
            Instances testingData = loadCSVData(testingFile);
            
            // Display basic dataset information
            System.out.println("\nTraining data loaded: " + trainingData.numInstances() + 
                              " instances with " + trainingData.numAttributes() + " attributes");
            System.out.println("Training data attributes: " + trainingData.toString().split("\n")[0]);
            
            // Convert the output (class) attribute to nominal
            trainingData = convertOutputToNominal(trainingData);
            testingData = convertOutputToNominal(testingData);
            
            // Set the class index to the Output column (last column)
            int classIndex = trainingData.numAttributes() - 1;
            trainingData.setClassIndex(classIndex);
            testingData.setClassIndex(classIndex);
            
            System.out.println("\nClass attribute: " + trainingData.classAttribute().name());
            System.out.println("Class values: " + trainingData.classAttribute().toString());
            
            // Analyze class distribution
            analyzeClassDistribution(trainingData);
            
            // Configure and build the J48 decision tree
            J48 j48 = new J48();
            
            // Configure J48 parameters - these can be tuned for optimal performance
            String[] options = new String[6];
            options[0] = "-C"; options[1] = "0.25";  // Confidence factor for pruning
            options[2] = "-M"; options[3] = "2";     // Minimum instances per leaf
            options[4] = "-A";                       // Don't perform subtree raising
            options[5] = "-B";                       // Use binary splits for nominal attributes
            j48.setOptions(options);
            
            System.out.println("\nBuilding J48 model with options: " + Utils.joinOptions(options));
            j48.buildClassifier(trainingData);
            
            // Print the resulting decision tree
            System.out.println("\nJ48 Decision Tree Model:");
            System.out.println(j48);
            
            // Evaluate with cross-validation on training data
            System.out.println("\nEvaluating with 10-fold cross-validation on training data...");
            Evaluation evalTrain = new Evaluation(trainingData);
            evalTrain.crossValidateModel(j48, trainingData, 10, new Random(1));
            printEvaluationResults(evalTrain);
            
            // Evaluate on the separate test set
            System.out.println("\nEvaluating on separate test data...");
            Evaluation evalTest = new Evaluation(trainingData);
            evalTest.evaluateModel(j48, testingData);
            printEvaluationResults(evalTest);
            
            // Make and display predictions for test instances
            System.out.println("\nSample predictions from test data:");
            System.out.println("----------------------------------------");
            
            // Show all predictions if test set is small, otherwise just show first 10
            int predictionsToShow = Math.min(10, testingData.numInstances());
            
            for (int i = 0; i < predictionsToShow; i++) {
                double actualValue = testingData.instance(i).classValue();
                double predictedValue = j48.classifyInstance(testingData.instance(i));
                
                System.out.println("Instance " + (i+1) + ":");
                System.out.println("  Open: " + testingData.instance(i).value(0));
                System.out.println("  High: " + testingData.instance(i).value(1));
                System.out.println("  Low: " + testingData.instance(i).value(2));
                System.out.println("  Close: " + testingData.instance(i).value(3));
                System.out.println("  Adj Close: " + testingData.instance(i).value(4));
                System.out.println("  Actual class: " + testingData.classAttribute().value((int)actualValue));
                System.out.println("  Predicted class: " + testingData.classAttribute().value((int)predictedValue));
                
                // Get confidence/probability of prediction
                double[] distribution = j48.distributionForInstance(testingData.instance(i));
                System.out.println("  Confidence: " + String.format("%.2f%%", distribution[(int)predictedValue] * 100));
                System.out.println("----------------------------------------");
            }
            
            // Optionally save the model for future use
            // weka.core.SerializationHelper.write("crypto_j48_model.model", j48);
            // System.out.println("Model saved to crypto_j48_model.model");
            
        } catch (Exception e) {
            System.err.println("Error in cryptocurrency prediction process:");
            e.printStackTrace();
        }
    }
    
    /**
     * Load CSV data from file
     */
    private static Instances loadCSVData(String csvFilePath) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvFilePath));
        return loader.getDataSet();
    }
    
    /**
     * Convert numeric Output column to nominal (required for classification)
     */
    private static Instances convertOutputToNominal(Instances data) throws Exception {
        // Convert the class attribute (Output) from numeric to nominal
        NumericToNominal convert = new NumericToNominal();
        
        // Set options to convert only the last attribute (Output)
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = Integer.toString(data.numAttributes()); // Last attribute index
        convert.setOptions(options);
        
        convert.setInputFormat(data);
        return Filter.useFilter(data, convert);
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
    
    /**
     * Print detailed evaluation results
     */
    private static void printEvaluationResults(Evaluation eval) throws Exception {
        System.out.println(eval.toSummaryString("\nResults:", false));
        System.out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");
        System.out.println("Precision: " + String.format("%.4f", eval.weightedPrecision()));
        System.out.println("Recall: " + String.format("%.4f", eval.weightedRecall()));
        System.out.println("F1 Score: " + String.format("%.4f", eval.weightedFMeasure()));
        
        System.out.println("\nConfusion Matrix:");
        System.out.println(eval.toMatrixString());
        
        // Detailed accuracy by class
        System.out.println("\nDetailed Accuracy by Class:");
        System.out.println(eval.toClassDetailsString());
    }
}
