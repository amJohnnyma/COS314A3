package com;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.ArrayList;

import java.awt.event.*;
import java.io.File;


public class SmoothingSwingTool extends JFrame {

    private TrainingMetrics metrics;
    private JLabel statusLabel;
    private JSlider smoothingSlider;
    private JButton loadButton, generateButton;
    private File selectedFile;

    public SmoothingSwingTool() {
        super("Smoothing Tool");

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 400);
        setLayout(new BorderLayout());

        JPanel topPanel = new JPanel();
        loadButton = new JButton("Load Metrics JSON");
        topPanel.add(loadButton);

        smoothingSlider = new JSlider(1, 200, 15);
                smoothingSlider.setPreferredSize(new Dimension(600, 200));

        smoothingSlider.setMajorTickSpacing(20);
        smoothingSlider.setPaintTicks(true);
        smoothingSlider.setPaintLabels(true);
        topPanel.add(new JLabel("Smoothing Window:"));
        topPanel.add(smoothingSlider);

        generateButton = new JButton("Generate Chart");
        generateButton.setEnabled(false);
        topPanel.add(generateButton);

        add(topPanel, BorderLayout.CENTER);

        statusLabel = new JLabel("Load a metrics JSON file to start.");
        add(statusLabel, BorderLayout.SOUTH);

        // Button listeners
        loadButton.addActionListener(e -> loadMetrics());
        generateButton.addActionListener(e -> generateChart());

        setVisible(true);
    }

    private void loadMetrics() {
        JFileChooser fileChooser = new JFileChooser(System.getProperty("user.dir") + "/myapp/WithStops/");
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

    private void generateChart() {
        if (metrics == null) {
            statusLabel.setText("No metrics loaded!");
            return;
        }
        int downsampleFactor = 50;  // Or get this from a slider/input to let user control it
        int smoothingWindow = smoothingSlider.getValue();  // Your existing smoothing window
        statusLabel.setText("Generating chart with smoothing window = " + smoothingWindow + "...");


        // Downsample first
        List<Double> dsLosses = TrainingMetrics.downsample(metrics.losses, downsampleFactor);
        List<Double> dsWeights = TrainingMetrics.downsample(metrics.avgWeights, downsampleFactor);
        List<Double> dsBiases = TrainingMetrics.downsample(metrics.avgBiases, downsampleFactor);
        List<Double> dsDeltaValues = TrainingMetrics.downsample(metrics.deltaValues, downsampleFactor);

        // For epoch times (Vector<Long>), convert to List<Double> first, then downsample:
        List<Double> epochTimesSeconds = metrics.epochTimes.stream().map(t -> t / 1000.0).toList();
        List<Double> dsEpochTimes = TrainingMetrics.downsample(epochTimesSeconds, downsampleFactor);

        // Now smooth the downsampled data:
        List<Double> smoothLosses = TrainingMetrics.smooth(dsLosses, smoothingWindow);
        List<Double> smoothWeights = TrainingMetrics.smooth(dsWeights, smoothingWindow);
        List<Double> smoothBiases = TrainingMetrics.smooth(dsBiases, smoothingWindow);
        List<Double> smoothEpochTimes = TrainingMetrics.smooth(dsEpochTimes, smoothingWindow);
        List<Double> smoothDeltas = TrainingMetrics.smooth(dsDeltaValues, smoothingWindow);


        Graph g = new Graph(smoothLosses, smoothWeights, smoothBiases, smoothEpochTimes, smoothDeltas);

        String originalName = selectedFile.getName();  // just the file name, no path
        String fileName = "SMOOTH: " + smoothingWindow + " - " + originalName + ".png";
        g.createChart(fileName);

        statusLabel.setText("Chart saved as: " + fileName);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(SmoothingSwingTool::new);
    }
}
