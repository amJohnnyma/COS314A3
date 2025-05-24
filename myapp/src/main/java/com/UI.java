package com;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import java.awt.*;
import java.util.List;

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
    private static final String MODE_GP = "Run GP";
    private static final String MODE_MLP_TEST = "TestMLP";
    private static final String MODE_DT = "RunDT";

    private String[] functions = { MODE_SMOOTHING, MODE_MLP, MODE_GP };

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
        modePanel.add(createGPPanel(), MODE_GP);
        add(modePanel, BorderLayout.CENTER);

        statusLabel = new JLabel("Select a mode to begin.");
        add(statusLabel, BorderLayout.SOUTH);

        setVisible(true);
    }

    private JPanel createSmoothingPanel() {
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout(10, 10));

        JPanel topPanel = new JPanel();
        topPanel.setLayout(new BoxLayout(topPanel, BoxLayout.X_AXIS));
        topPanel.add(loadButton = new JButton("Load Metrics JSON"));
        topPanel.add(Box.createHorizontalStrut(15));
        topPanel.add(new JLabel("Smoothing Window:"));
        topPanel.add(Box.createHorizontalStrut(10));
        smoothingSlider = new JSlider(1, 200, 15);
        smoothingSlider.setMajorTickSpacing(20);
        smoothingSlider.setPaintTicks(true);
        smoothingSlider.setPaintLabels(true);
        topPanel.add(smoothingSlider);

        JPanel bottomPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 5));
        generateButton = new JButton("Generate Selected File");
        generateButton.setEnabled(false);
        processAllButton = new JButton("Process All Files");
        bottomPanel.add(generateButton);
        bottomPanel.add(processAllButton);

        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        mainPanel.add(topPanel, BorderLayout.NORTH);
        mainPanel.add(bottomPanel, BorderLayout.SOUTH);

        // Listeners
        loadButton.addActionListener(e -> loadMetrics());
        generateButton.addActionListener(e -> generateChart());
        processAllButton.addActionListener(e -> batchLoadMetrics());

        return mainPanel;
    }

    private JPanel createGPPanel() {
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout(10, 10));

        JPanel inputPanel = new JPanel();
        inputPanel.setLayout(new GridLayout(0, 2, 10, 10));

        JTextField maxNumGensField = new JTextField("100");
        JTextField populationSizeField = new JTextField("20");
        JTextField CrossoverRate = new JTextField("0.8");
        JTextField MutationRate = new JTextField("0.12");

        inputPanel.add(new JLabel("Maximum number of generations:"));
        inputPanel.add(maxNumGensField);
        inputPanel.add(new JLabel("Population size:"));
        inputPanel.add(populationSizeField);
        inputPanel.add(new JLabel("Crossover Rate:"));
        inputPanel.add(CrossoverRate);
        inputPanel.add(new JLabel("Mutation Rate:"));
        inputPanel.add(MutationRate);

        JButton runGPButton = new JButton("Run GP");
        inputPanel.add(runGPButton);
        inputPanel.add(new JLabel(""));

        JTextArea consoleArea = new JTextArea("Output");
        consoleArea.setLineWrap(true);
        consoleArea.setWrapStyleWord(true);
        JScrollPane scrollPane = new JScrollPane(consoleArea);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setPreferredSize(new Dimension(800, 300));

        mainPanel.add(inputPanel, BorderLayout.NORTH);
        mainPanel.add(scrollPane, BorderLayout.CENTER);

        runGPButton.addActionListener(e -> {
            try {
                int maxGenNum = Integer.parseInt(maxNumGensField.getText());
                int popSize = Integer.parseInt(populationSizeField.getText());
                double COrate = Double.parseDouble(CrossoverRate.getText());
                double MutationRa = Double.parseDouble(MutationRate.getText());

                GP gp = new GP(maxGenNum, popSize,COrate, MutationRa,
                        System.getProperty("user.dir") + "/myapp/src/data/BTC_train.csv",
                        System.getProperty("user.dir") + "/myapp/src/data/BTC_test.csv");
                boolean hundred = false;
                Individual bestIndividual = null;
                while (!hundred) {
                    gp.Algorithm();
                    bestIndividual = gp.getBestIndividual();
                    if(gp.GetAccuracyOfBestIndividual() == 100){
                        hundred = true;
                    }
                }

                consoleArea.setText(
                        "Best Individual: " + bestIndividual.toString() +
                                "\nFitness: " + bestIndividual.fitness +
                                "\nAccuracy: " + gp.GetAccuracyOfBestIndividual());
            } catch (Exception ex) {
                consoleArea.setText("Error: " + ex.getMessage());
                ex.printStackTrace();
            }
        });

        return mainPanel;
    }

    private JPanel createMLPPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(0, 2, 10, 10));

        JTextField iterationsField = new JTextField("1000");
        JTextField runsField = new JTextField("10");
        JTextField learningRateField = new JTextField("0.01");
        JTextField seedField = new JTextField("0");
        JTextField batchSizeField = new JTextField("16");
        JTextField hiddenSizeField = new JTextField("32");
        JTextField hiddenLayersField = new JTextField("2");
        JTextField targetAccField = new JTextField("0.95");
        JTextField patienceField = new JTextField("50");
        JTextField minImproField = new JTextField("0.01");

        JButton runMLPButton = new JButton("Run MLP");
        JButton testMLPButton = new JButton("Test MLP");
        JButton loadMLPButton = new JButton("Load MLP");

        JTextArea consoleArea = new JTextArea("Output");
        consoleArea.setLineWrap(true);
        consoleArea.setWrapStyleWord(true);
        JScrollPane scrollPane = new JScrollPane(consoleArea);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);

        loadMLPButton.addActionListener(e -> {

            loadMLP();
            if (selectedFile.exists()) {
                testMLPButton.setEnabled(true);

            }
        });

        runMLPButton.addActionListener(e -> {
            try {
                int iterations = Integer.parseInt(iterationsField.getText());
                double learningRate = Double.parseDouble(learningRateField.getText());
                long seed = Long.parseLong(seedField.getText());
                int batchSize = Integer.parseInt(batchSizeField.getText());
                int hiddenSize = Integer.parseInt(hiddenSizeField.getText());
                int hiddenLayers = Integer.parseInt(hiddenLayersField.getText());
                double targetAccuracy = Double.parseDouble(targetAccField.getText());
                int runs = Integer.parseInt(runsField.getText());
                double minImpro = Double.parseDouble(minImproField.getText());
                int patience = Integer.parseInt(patienceField.getText());

                AlgoFunctions af = new AlgoFunctions();
                statusLabel.setText("MLP training started...");

                List<String> out = af.runMLP(iterations, runs, learningRate, seed, batchSize, hiddenSize, hiddenLayers,
                        targetAccuracy, patience, minImpro);
                if (!out.isEmpty()) {
                    statusLabel.setText("MLP training finished");
                    consoleArea.setText(String.join("\n\n", out));
                } else {
                    statusLabel.setText("MLP training started...");
                }

            } catch (Exception ex) {
                statusLabel.setText("Invalid MLP parameter input.");
                ex.printStackTrace();
            }
        });
        testMLPButton.setEnabled(false);
        testMLPButton.addActionListener(e -> {
            AlgoFunctions af = new AlgoFunctions();
            consoleArea.setText(af.testMLP(selectedFile.getName()));
        });

        panel.add(new JLabel("Iterations(int):"));
        panel.add(iterationsField);
        panel.add(new JLabel("Runs(int):"));
        panel.add(runsField);
        panel.add(new JLabel("Learning Rate(double):"));
        panel.add(learningRateField);
        panel.add(new JLabel("Seed(long):"));
        panel.add(seedField);
        panel.add(new JLabel("Batch Size(int):"));
        panel.add(batchSizeField);
        panel.add(new JLabel("Hidden Size(int):"));
        panel.add(hiddenSizeField);
        panel.add(new JLabel("Hidden Layers(int):"));
        panel.add(hiddenLayersField);
        panel.add(new JLabel("Target Accuracy(double):"));
        panel.add(targetAccField);
        panel.add(new JLabel("Patience(int):"));
        panel.add(patienceField);
        panel.add(new JLabel("Min improvement(double):"));
        panel.add(minImproField);
        panel.add(runMLPButton);
        panel.add(testMLPButton);
        panel.add(loadMLPButton);
        panel.add(scrollPane);

        return panel;
    }

    private void batchLoadMetrics() {
        File dir = new File(System.getProperty("user.dir") + "/WithStops/");
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
            statusLabel.setText("No .json files found in directory.");
        }
    }

    private void loadMetrics() {
        JFileChooser fileChooser = new JFileChooser(System.getProperty("user.dir") + "/WithStops/");
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            selectedFile = fileChooser.getSelectedFile();
            metrics = TrainingMetrics.loadFromFile(selectedFile.getAbsolutePath());
            if (metrics != null) {
                statusLabel.setText("Loaded: " + selectedFile.getName());
                generateButton.setEnabled(true);
            } else {
                statusLabel.setText("Failed to load metrics file.");
                generateButton.setEnabled(false);
            }
        }
    }

    private void loadMLP() {
        JFileChooser fileChooser = new JFileChooser(System.getProperty("user.dir"));

        // Set to only allow file selection (not directories)
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        // Filter for .mlp files
        FileNameExtensionFilter filter = new FileNameExtensionFilter("MLP Model Files", "mlp");
        fileChooser.setFileFilter(filter);

        // Show dialog
        int result = fileChooser.showOpenDialog(null);
        if (result == JFileChooser.APPROVE_OPTION) {
            selectedFile = fileChooser.getSelectedFile();
        }
    }

    private void generateChart() {
        if (metrics == null) {
            statusLabel.setText("No metrics loaded!");
            return;
        }
        int downsampleFactor = 50; // Or get this from a slider/input to let user control it
        int smoothingWindow = smoothingSlider.getValue(); // Your existing smoothing window
        statusLabel.setText("Generating chart with smoothing window = " + smoothingWindow + "...");

        // Downsample first
        List<Double> dsLosses = TrainingMetrics.downsample(metrics.losses, downsampleFactor);
        List<Double> dsWeights = TrainingMetrics.downsample(metrics.avgWeights, downsampleFactor);
        List<Double> dsBiases = TrainingMetrics.downsample(metrics.avgBiases, downsampleFactor);
        List<Double> dsDeltaValues = TrainingMetrics.downsample(metrics.deltaValues, downsampleFactor);

        // For epoch times (Vector<Long>), convert to List<Double> first, then
        // downsample:
        List<Double> epochTimesSeconds = metrics.epochTimes.stream().map(t -> t / 1000.0).toList();
        List<Double> dsEpochTimes = TrainingMetrics.downsample(epochTimesSeconds, downsampleFactor);

        // Now smooth the downsampled data:
        List<Double> smoothLosses = TrainingMetrics.smooth(dsLosses, smoothingWindow);
        List<Double> smoothWeights = TrainingMetrics.smooth(dsWeights, smoothingWindow);
        List<Double> smoothBiases = TrainingMetrics.smooth(dsBiases, smoothingWindow);
        List<Double> smoothEpochTimes = TrainingMetrics.smooth(dsEpochTimes, smoothingWindow);
        List<Double> smoothDeltas = TrainingMetrics.smooth(dsDeltaValues, smoothingWindow);

        Graph g = new Graph(smoothLosses, smoothWeights, smoothBiases, smoothEpochTimes, smoothDeltas);

        String originalName = selectedFile.getName(); // just the file name, no path
        String fileName = "SMOOTH: " + smoothingWindow + " - " + originalName + ".png";
        g.createChart(fileName);

        statusLabel.setText("Chart saved as: " + fileName);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(UI::new);
    }

}
