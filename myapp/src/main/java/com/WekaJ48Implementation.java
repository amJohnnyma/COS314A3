package com;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Normalize;
import weka.core.Attribute;
import java.util.ArrayList;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.evaluation.Evaluation;
import java.io.File;
import java.util.Random;

public class WekaJ48Implementation {
    
    public void DT() {
        try {
            System.out.println("Starting Weka J48 implementation...");
            
            // Define file paths (adjust these to your actual file locations)
            String trainingFilePath = "src/data/BTC_train.csv";
            String testingFilePath = "src/data/BTC_test.csv";
            
            // Step 1: Load training data
            Instances trainingData = loadDataFromCSV(trainingFilePath);
            if (trainingData == null) return;

            // inspectData(trainingData, "Before PreProcessing");
            // Step 2: Preprocess the data for J48
            trainingData = preprocessData(trainingData);
            
            // inspectData(trainingData, "After PreProcessing");
            // Step 3: Build J48 model
            J48 j48Tree = buildJ48Model(trainingData);
            if (j48Tree == null) return;
            
            // Step 4: Load and preprocess testing data
            Instances testingData = loadDataFromCSV(testingFilePath);
            if (testingData == null) return;
            
            // Make sure test data structure matches training data
            testingData.setClassIndex(testingData.numAttributes() - 1);
            
            // Apply same preprocessing to test data
            testingData = preprocessTestData(testingData, trainingData.classAttribute().isNominal());
            
            // Step 5: Evaluate model on test data instead of just making predictions
            evaluateModelOnTestData(j48Tree, testingData);
            
            System.out.println("J48 implementation completed successfully!");
            
        } catch (Exception e) {
            System.err.println("An error occurred during execution:");
            e.printStackTrace();
        }
    }
    private static void inspectData(Instances data, String stage) {
    System.out.println("\n=== Data Inspection (" + stage + ") ===");
    System.out.println("Number of instances: " + data.numInstances());
    System.out.println("Number of attributes: " + data.numAttributes());
    
    // Print first few instances
    System.out.println("\nFirst 5 instances:");
    for (int i = 0; i < Math.min(5, data.numInstances()); i++) {
        System.out.println(data.instance(i));
    }
    
    // Print class attribute information if set
    if (data.classIndex() >= 0) {
        System.out.println("\nClass attribute: " + data.classAttribute().name());
        System.out.println("Class type: " + (data.classAttribute().isNominal() ? "Nominal" : "Numeric"));
    }
    
    // Print basic statistics
    System.out.println("\nBasic statistics for numeric attributes:");
    for (int i = 0; i < data.numAttributes(); i++) {
        if (data.attribute(i).isNumeric()) {
            System.out.println(String.format("\n%s:", data.attribute(i).name()));
            System.out.println(String.format("  Min: %.3f", data.attributeStats(i).numericStats.min));
            System.out.println(String.format("  Max: %.3f", data.attributeStats(i).numericStats.max));
            System.out.println(String.format("  Mean: %.3f", data.attributeStats(i).numericStats.mean));
            System.out.println(String.format("  StdDev: %.3f", data.attributeStats(i).numericStats.stdDev));
        }
    }
    System.out.println("\n" + "=".repeat(50));
}
    private static Instances loadDataFromCSV(String filePath) {
        try {
            System.out.println("Loading data from: " + filePath);
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filePath));
            return loader.getDataSet();
            } 
            catch (Exception e) {
                System.err.println("Error loading data from CSV file: " + filePath);
                e.printStackTrace();
                return null;
            }
        }

        private static Instances preprocessData(Instances data) throws Exception {
        System.out.println("Preprocessing training data...");
        
        // Set class index first
        data.setClassIndex(data.numAttributes() - 1);
        
        // Add technical indicators
        data = addTechnicalIndicators(data);
        
        // Normalize numeric attributes
        Normalize normalize = new Normalize();
        normalize.setInputFormat(data);
        data = Filter.useFilter(data, normalize);

        // Convert ALL numeric attributes to nominal using discretization
        NumericToNominal numToNom = new NumericToNominal();
        String rangeList = "1-" + data.numAttributes(); // Convert all attributes
        numToNom.setOptions(new String[]{"-R", rangeList});
        numToNom.setInputFormat(data);
        data = Filter.useFilter(data, numToNom);
        
        // Use supervised discretization instead of NumericToNominal
        weka.filters.supervised.attribute.Discretize discretize = new weka.filters.supervised.attribute.Discretize();
        discretize.setOptions(new String[]{
            "-R", "first-last",    // Apply to all attributes except class
            "-precision", "6",     // Precision of cut points
            "-Y"                   // Don't discretize the class attribute
        });
        discretize.setInputFormat(data);
        data = Filter.useFilter(data, discretize);

        System.out.println("Discretized numeric attributes into bins");

        // balanceClasses(data);

        // Feature selection
        AttributeSelection attSelect = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();
        search.setThreshold(0.01);
        attSelect.setEvaluator(eval);
        attSelect.setSearch(search);
        attSelect.SelectAttributes(data);
        data = attSelect.reduceDimensionality(data);

        return data;
    }


        // private static Instances balanceClasses(Instances data) throws Exception {
        //     System.out.println("Balancing classes with corrected Resample filter...");
            
        //     // Find minority and majority class counts
        //     int[] classCounts = new int[data.numClasses()];
        //     for (int i = 0; i < data.numInstances(); i++) {
        //         classCounts[(int)data.instance(i).classValue()]++;
        //     }
            
        //     // Find minority class
        //     int minorityClass = 0;
        //     for (int i = 1; i < classCounts.length; i++) {
        //         if (classCounts[i] < classCounts[minorityClass]) {
        //             minorityClass = i;
        //         }
        //     }
            
        //     // Prepare options for biased sampling
        //     String biasToMinorityClass = String.valueOf(minorityClass);
            
        //     Resample resample = new Resample();
        //     resample.setOptions(new String[] {
        //         "-B", "1.0",           // Maintain dataset size
        //         "-S", "1",             // Random seed
        //         "-Z", "100",           // Size of output dataset as percentage
        //         "-W",                  // Enable biased sampling  
        //         "-c", biasToMinorityClass  // Class index to bias towards
        //     });
            
        //     resample.setInputFormat(data);
        //     return Filter.useFilter(data, resample);
        // }

        private static Instances addTechnicalIndicators(Instances data) throws Exception {
        // Create arrays to store price data
        double[] closes = new double[data.numInstances()];
        double[] highs = new double[data.numInstances()];
        double[] lows = new double[data.numInstances()];
        
        // Extract price data
        for (int i = 0; i < data.numInstances(); i++) {
            closes[i] = data.instance(i).value(3); // Close price
            highs[i] = data.instance(i).value(1);  // High price
            lows[i] = data.instance(i).value(2);   // Low price
        }

        // Create new attributes list
        ArrayList<Attribute> newAttributes = new ArrayList<>();

        // Copy existing attributes
        for (int i = 0; i < data.numAttributes(); i++) {
            newAttributes.add(data.attribute(i));
        }

        // Add new technical indicators
        newAttributes.add(new Attribute("RSI_14"));
        newAttributes.add(new Attribute("SMA_5"));
        newAttributes.add(new Attribute("SMA_20"));
        newAttributes.add(new Attribute("Price_ROC"));
        newAttributes.add(new Attribute("TR"));

        // Create new dataset
        Instances newData = new Instances("enhanced_" + data.relationName(), 
                                        newAttributes, 
                                        data.numInstances());
        newData.setClassIndex(data.classIndex());

        // Calculate indicators
        for (int i = 0; i < data.numInstances(); i++) {
            double[] values = new double[newAttributes.size()];

            // Copy existing values
            for (int j = 0; j < data.numAttributes(); j++) {
                values[j] = data.instance(i).value(j);
            }

            // Add technical indicators
            int idx = data.numAttributes();
            values[idx++] = calculateRSI(closes, i, 14);
            values[idx++] = calculateSMA(closes, i, 5);
            values[idx++] = calculateSMA(closes, i, 20);
            values[idx++] = calculateROC(closes, i, 14);
            values[idx++] = calculateTR(highs[i], lows[i], i > 0 ? closes[i-1] : closes[i]);

            newData.add(new weka.core.DenseInstance(1.0, values));
        }

        return newData;
    }

    private static double calculateRSI(double[] prices, int currentIndex, int period) {
        if (currentIndex < period) return 50;

        double sumGain = 0, sumLoss = 0;
        for (int i = currentIndex - period + 1; i <= currentIndex; i++) {
            double diff = i > 0 ? prices[i] - prices[i-1] : 0;
            if (diff > 0) sumGain += diff;
            else sumLoss -= diff;
        }

        if (sumLoss == 0) return 100;
        double rs = sumGain / sumLoss;
        return 100 - (100 / (1 + rs));
    }

    private static double calculateSMA(double[] prices, int currentIndex, int period) {
        if (currentIndex < period) return prices[currentIndex];

        double sum = 0;
        for (int i = currentIndex - period + 1; i <= currentIndex; i++) {
            sum += prices[i];
        }
        return sum / period;
    }

    private static double calculateROC(double[] prices, int currentIndex, int period) {
        if (currentIndex < period) return 0;
        return ((prices[currentIndex] - prices[currentIndex - period]) / 
                prices[currentIndex - period]) * 100;
    }

    private static double calculateTR(double high, double low, double prevClose) {
        return Math.max(high - low, 
               Math.max(Math.abs(high - prevClose), 
                       Math.abs(low - prevClose)));
    }

    private static Instances preprocessTestData(Instances testData, boolean isClassNominal) throws Exception 
    {
        System.out.println("Preprocessing test data...");
        
        // Set class index first
        testData.setClassIndex(testData.numAttributes() - 1);
        
        // Add same technical indicators as training data
        testData = addTechnicalIndicators(testData);
        
        // Apply same normalization
        Normalize normalize = new Normalize();
        normalize.setInputFormat(testData);
        testData = Filter.useFilter(testData, normalize);

        // Convert ALL numeric attributes to nominal using discretization
        NumericToNominal numToNom = new NumericToNominal();
        String rangeList = "1-" + testData.numAttributes(); // Convert all attributes
        numToNom.setOptions(new String[]{"-R", rangeList});
        numToNom.setInputFormat(testData);
        testData = Filter.useFilter(testData, numToNom);

        // Apply same discretization
        weka.filters.supervised.attribute.Discretize discretize = new weka.filters.supervised.attribute.Discretize();
        discretize.setOptions(new String[]{
            "-R", "first-last",    // Apply to all attributes except class
            "-precision", "6",     // Precision of cut points
            "-Y"                   // Don't discretize the class attribute
        });
        discretize.setInputFormat(testData);
        testData = Filter.useFilter(testData, discretize);

        return testData;
    }
    
    private static J48 buildJ48Model(Instances trainingData) {
        try {
            System.out.println("Building J48 decision tree model...");
            
            // Create J48 classifier
            J48 j48Tree = new J48();
            
            // Configure J48 parameters
            j48Tree.setOptions(new String[]
            {
                "-C","0.25" , // Make unpruned
                "-M", "5",    // Minimum instances per leaf
                "-A",         // Laplace smoothing
                "-B"          // Use reduced error pruning
            });
            
            // Build the classifier
            j48Tree.buildClassifier(trainingData);
            
            // Output model information
            System.out.println("\nJ48 Decision Tree Model:");
            System.out.println(j48Tree.toString());
            
            // Evaluate model on training data
            evaluateModelOnTrainingData(j48Tree, trainingData);
            
            return j48Tree;
            
        } catch (Exception e) {
            System.err.println("Error building J48 model:");
            e.printStackTrace();
            return null;
        }
    }
    
    private static void evaluateModelOnTrainingData(J48 model, Instances data) {
        try {
            System.out.println("\nEvaluating model on training data (cross-validation)...");
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));
            
            printEvaluationMetrics(eval, "TRAINING EVALUATION (Cross-validation)");
            
        } catch (Exception e) {
            System.err.println("Error evaluating model on training data:");
            e.printStackTrace();
        }
    }
    
    private static void evaluateModelOnTestData(J48 model, Instances testData) {
        try {
            System.out.println("\nEvaluating model on test data...");
            
            Evaluation eval = new Evaluation(testData);
            eval.evaluateModel(model, testData);
            
            printEvaluationMetrics(eval, "TEST DATA EVALUATION");
            
        } catch (Exception e) {
            System.err.println("Error evaluating model on test data:");
            e.printStackTrace();
        }
    }
    
    private static void printEvaluationMetrics(Evaluation eval, String title) {
    try {
        System.out.println("\n=== " + title + " ===");
        System.out.println("Accuracy: " + String.format("%.2f%%", eval.pctCorrect()));
        
        System.out.println("\nDetailed Accuracy By Class:");
        System.out.println("--------------------------------------------------");
        System.out.println("                TP Rate  FP Rate  Precision  Recall   F-Measure  Class");
        
        // Get number of classes from the confusion matrix
        double[][] confusionMatrix = eval.confusionMatrix();
        int numClasses = confusionMatrix.length;
        
        // Calculate and print metrics for each class
        for (int i = 0; i < numClasses; i++) {
            double precision = eval.precision(i);
            double recall = eval.recall(i);
            double fMeasure = eval.fMeasure(i);
            double tpRate = eval.truePositiveRate(i);
            double fpRate = eval.falsePositiveRate(i);
            
            System.out.println(String.format("%-15s %.3f    %.3f    %.3f      %.3f    %.3f      %d", 
                              "", tpRate, fpRate, precision, recall, fMeasure, i));
        }
        
        // Print weighted averages
        System.out.println("--------------------------------------------------");
        System.out.println(String.format("Weighted Avg.    %.3f    %.3f    %.3f      %.3f    %.3f", 
                          eval.weightedTruePositiveRate(), 
                          eval.weightedFalsePositiveRate(),
                          eval.weightedPrecision(),
                          eval.weightedRecall(),
                          eval.weightedFMeasure()));
        
        // Print confusion matrix
        System.out.println("\nConfusion Matrix:");
        System.out.println(eval.toMatrixString());
    } catch (Exception e) {
        System.err.println("Error printing evaluation metrics:");
        e.printStackTrace();
    }
}
}