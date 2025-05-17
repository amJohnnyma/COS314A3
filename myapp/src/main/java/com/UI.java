package com;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.ArrayList;

import java.io.File;

public class UI extends JFrame {

    private TrainingMetrics metrics;
    private JLabel statusLabel;
    private JSlider smoothingSlider;
    private JButton loadButton, generateButton, processAllButton;
    private File selectedFile;
    // At the top of your class
    private JPanel modePanel;
    private CardLayout cardLayout;

    // Function modes
    private static final String MODE_SMOOTHING = "Smoothing Graphs";
    private static final String MODE_MLP = "RunMLP";
    private static final String MODE_DT = "RunDT";

    private String[] functions = { MODE_SMOOTHING, MODE_MLP };

    public UI() {
        super("COS314A3");

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1000, 600);
        setLayout(new BorderLayout());

        // Mode selector
        JComboBox<String> funcSelector = new JComboBox<>(functions);
        funcSelector.addActionListener(e -> cardLayout.show(modePanel, (String) funcSelector.getSelectedItem()));

        JPanel topBar = new JPanel();
        topBar.add(new JLabel("Mode:"));
        topBar.add(funcSelector);
        add(topBar, BorderLayout.NORTH);

        // Card layout to switch between function modes
        cardLayout = new CardLayout();
        modePanel = new JPanel(cardLayout);

        modePanel.add(createSmoothingPanel(), MODE_SMOOTHING);
        modePanel.add(createMLPPanel(), MODE_MLP);
        add(modePanel, BorderLayout.CENTER);

        statusLabel = new JLabel("Select a mode to begin.");
        add(statusLabel, BorderLayout.SOUTH);

        setVisible(true);
    }

    private JPanel createSmoothingPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        loadButton = new JButton("Load Metrics JSON");
        generateButton = new JButton("Generate Selected File");
        generateButton.setEnabled(false);
        processAllButton = new JButton("Process all Files");

        smoothingSlider = new JSlider(1, 200, 15);
        smoothingSlider.setMajorTickSpacing(20);
        smoothingSlider.setPaintTicks(true);
        smoothingSlider.setPaintLabels(true);

        panel.add(loadButton);
        panel.add(new JLabel("Smoothing Window:"));
        panel.add(smoothingSlider);
        panel.add(generateButton);
        panel.add(processAllButton);

        // Listeners
        loadButton.addActionListener(e -> loadMetrics());
        generateButton.addActionListener(e -> generateChart());
        processAllButton.addActionListener(e -> batchLoadMetrics());

        return panel;
    }

    private JPanel createMLPPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(0, 2, 10, 10));

        JTextField iterationsField = new JTextField("10");
        JTextField learningRateField = new JTextField("0.01");
        JTextField seedField = new JTextField("123");
        JTextField batchSizeField = new JTextField("32");
        JTextField hiddenSizeField = new JTextField("64");
        JTextField hiddenLayersField = new JTextField("2");
        JTextField targetAccField = new JTextField("0.95");

        JButton runMLPButton = new JButton("Run MLP");

        runMLPButton.addActionListener(e -> {
            try {
                int iterations = Integer.parseInt(iterationsField.getText());
                double learningRate = Double.parseDouble(learningRateField.getText());
                long seed = Long.parseLong(seedField.getText());
                int batchSize = Integer.parseInt(batchSizeField.getText());
                int hiddenSize = Integer.parseInt(hiddenSizeField.getText());
                int hiddenLayers = Integer.parseInt(hiddenLayersField.getText());
                double targetAccuracy = Double.parseDouble(targetAccField.getText());

                AlgoFunctions af = new AlgoFunctions();
                af.runMLP(iterations, 1, learningRate, seed, batchSize, hiddenSize, hiddenLayers, targetAccuracy);

                statusLabel.setText("MLP training started...");

            } catch (Exception ex) {
                statusLabel.setText("Invalid MLP parameter input.");
                ex.printStackTrace();
            }
        });

        panel.add(new JLabel("Iterations:"));
        panel.add(iterationsField);
        panel.add(new JLabel("Learning Rate:"));
        panel.add(learningRateField);
        panel.add(new JLabel("Seed:"));
        panel.add(seedField);
        panel.add(new JLabel("Batch Size:"));
        panel.add(batchSizeField);
        panel.add(new JLabel("Hidden Size:"));
        panel.add(hiddenSizeField);
        panel.add(new JLabel("Hidden Layers:"));
        panel.add(hiddenLayersField);
        panel.add(new JLabel("Target Accuracy:"));
        panel.add(targetAccField);
        panel.add(runMLPButton);

        return panel;
    }

    private void batchLoadMetrics() {
        // Get the application's current directory
        File currentDir = new File(System.getProperty("user.dir"));
        File dir = new File(currentDir, "WithStops");
        
        if (!dir.exists()) {
            statusLabel.setText("Directory not found: " + dir.getAbsolutePath());
            return;
        }
        
        File[] files = dir.listFiles((d, name) -> name.endsWith(".json"));

        if (files != null && files.length > 0) {
            int loadedCount = 0;

            for (File file : files) {
                TrainingMetrics loadedMetrics = TrainingMetrics.loadFromFile(file.getAbsolutePath());
                if (loadedMetrics != null) {
                    metrics = loadedMetrics; // store last loaded metrics if needed
                    selectedFile = file;
                    generateChart(); // use the selectedFile and metrics
                    loadedCount++;
                } else {
                    System.err.println("Failed to load: " + file.getName());
                }
            }

            statusLabel.setText("Loaded and processed " + loadedCount + " files.");
        } else {
            statusLabel.setText("No .json files found in directory: " + dir.getAbsolutePath());
        }
    }

    private void loadMetrics() {
        // Start in the current directory or the WithStops directory if it exists
        File currentDir = new File(System.getProperty("user.dir"));
        File withStopsDir = new File(currentDir, "WithStops");
        File startDir = withStopsDir.exists() ? withStopsDir : currentDir;
        
        JFileChooser fileChooser = new JFileChooser(startDir);
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            selectedFile = fileChooser.getSelectedFile();
            statusLabel.setText("Loading: " + selectedFile.getName() + "...");
            
            try {
                metrics = TrainingMetrics.loadFromFile(selectedFile.getAbsolutePath());
                if (metrics != null) {
                    statusLabel.setText("Loaded: " + selectedFile.getName());
                    generateButton.setEnabled(true);
                } else {
                    statusLabel.setText("Failed to load metrics file.");
                    generateButton.setEnabled(false);
                }
            } catch (Exception e) {
                statusLabel.setText("Error loading file: " + e.getMessage());
                e.printStackTrace();
                generateButton.setEnabled(false);
            }
        }
    }

    private void generateChart() {
        if (metrics == null) {
            statusLabel.setText("No metrics loaded!");
            return;
        }
        
        try {
            int downsampleFactor = 50; // Or get this from a slider/input to let user control it
            int smoothingWindow = smoothingSlider.getValue(); // Your existing smoothing window
            statusLabel.setText("Generating chart with smoothing window = " + smoothingWindow + "...");
    
            // Check if metrics contain data
            if (metrics.losses == null || metrics.losses.isEmpty()) {
                statusLabel.setText("No loss data found in metrics!");
                return;
            }
    
            // Downsample first
            List<Double> dsLosses = TrainingMetrics.downsample(metrics.losses, downsampleFactor);
            
            List<Double> dsWeights = (metrics.avgWeights != null && !metrics.avgWeights.isEmpty()) ?
                    TrainingMetrics.downsample(metrics.avgWeights, downsampleFactor) : new ArrayList<>();
                    
            List<Double> dsBiases = (metrics.avgBiases != null && !metrics.avgBiases.isEmpty()) ?
                    TrainingMetrics.downsample(metrics.avgBiases, downsampleFactor) : new ArrayList<>();
                    
            List<Double> dsDeltaValues = (metrics.deltaValues != null && !metrics.deltaValues.isEmpty()) ?
                    TrainingMetrics.downsample(metrics.deltaValues, downsampleFactor) : new ArrayList<>();
    
            // For epoch times (Vector<Long>), convert to List<Double> first, then downsample:
            List<Double> epochTimesSeconds = new ArrayList<>();
            if (metrics.epochTimes != null && !metrics.epochTimes.isEmpty()) {
                for (Long time : metrics.epochTimes) {
                    epochTimesSeconds.add(time / 1000.0);
                }
            }
            List<Double> dsEpochTimes = (!epochTimesSeconds.isEmpty()) ?
                    TrainingMetrics.downsample(epochTimesSeconds, downsampleFactor) : new ArrayList<>();
    
            // Now smooth the downsampled data:
            List<Double> smoothLosses = TrainingMetrics.smooth(dsLosses, smoothingWindow);
            
            List<Double> smoothWeights = (!dsWeights.isEmpty()) ?
                    TrainingMetrics.smooth(dsWeights, smoothingWindow) : new ArrayList<>();
                    
            List<Double> smoothBiases = (!dsBiases.isEmpty()) ?
                    TrainingMetrics.smooth(dsBiases, smoothingWindow) : new ArrayList<>();
                    
            List<Double> smoothEpochTimes = (!dsEpochTimes.isEmpty()) ?
                    TrainingMetrics.smooth(dsEpochTimes, smoothingWindow) : new ArrayList<>();
                    
            List<Double> smoothDeltas = (!dsDeltaValues.isEmpty()) ?
                    TrainingMetrics.smooth(dsDeltaValues, smoothingWindow) : new ArrayList<>();
    
            Graph g = new Graph(smoothLosses, smoothWeights, smoothBiases, smoothEpochTimes, smoothDeltas);
    
            String originalName = selectedFile.getName(); // just the file name, no path
            String fileName = "SMOOTH_" + smoothingWindow + "_" + originalName + ".png";
            g.createChart(fileName);
    
            statusLabel.setText("Chart saved as: " + fileName);
        } catch (Exception e) {
            statusLabel.setText("Error generating chart: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(UI::new);
    }
}