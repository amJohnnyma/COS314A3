package com;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.awt.Dimension;


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
        int window = smoothingSlider.getValue();
        statusLabel.setText("Generating chart with smoothing window = " + window + "...");

        Graph g = new Graph(
            TrainingMetrics.smooth(metrics.losses, window),
            TrainingMetrics.smooth(metrics.avgWeights, window),
            TrainingMetrics.smooth(metrics.avgBiases, window),
            TrainingMetrics.smoothEpochTimes(metrics.epochTimes, window),
            TrainingMetrics.smooth(metrics.deltaValues, window)
        );

        String fileName = "Smoothed_" + window + ".png";
        g.createChart(fileName);

        statusLabel.setText("Chart saved as: " + fileName);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(SmoothingSwingTool::new);
    }
}
