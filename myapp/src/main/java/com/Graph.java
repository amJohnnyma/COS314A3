package com;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.util.Vector;

public class Graph {

    private Vector<Double> losses;
    private Vector<Double> avgWeights;
    private Vector<Double> avgBiases;
    private Vector<Long> epochTimes;

    public Graph(Vector<Double> losses, Vector<Double> avgWeights, Vector<Double> avgBiases, Vector<Long> epochTimes) {
        this.losses = losses;
        this.avgWeights = avgWeights;
        this.avgBiases = avgBiases;
        this.epochTimes = epochTimes;
    }

    public void createAndShowChart(String name) {
        // Create a dataset using the losses
        DefaultCategoryDataset dataset = createDataset();

        JFreeChart chart = ChartFactory.createLineChart(
                "Training Metrics Over Time",
                "Epoch",
                "Value",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );


        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));
        saveChartAsPNG(chart, name, 800, 600);

        // Create a panel to display the chart
        /*
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));

        // Create a frame to display the chart
        JFrame frame = new JFrame("Loss Graph");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(chartPanel);
        frame.pack();
        frame.setVisible(true);
        */
    }

    private DefaultCategoryDataset createDataset() {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (int i = 0; i < losses.size(); i++) {
            dataset.addValue(losses.get(i), "Loss", Integer.toString(i + 1));
            if (avgWeights != null && i < avgWeights.size())
                dataset.addValue(avgWeights.get(i), "Avg Weights", Integer.toString(i + 1));
            if (avgBiases != null && i < avgBiases.size())
                dataset.addValue(avgBiases.get(i), "Avg Biases", Integer.toString(i + 1));
            if (epochTimes != null && i < epochTimes.size())
                dataset.addValue(epochTimes.get(i) / 1000.0, "Epoch Time (s)", Integer.toString(i + 1));
        }

        return dataset;
    }

        private void saveChartAsPNG(JFreeChart chart, String fileName, int width, int height) {
        try {
            // Create a BufferedImage and draw the chart to it
            BufferedImage bufferedImage = chart.createBufferedImage(width, height);

            // Save the BufferedImage as a PNG file
            File directory = new File("WithStops");  // Create a directory called "charts" if it doesn't exist
            if (!directory.exists()) {
                directory.mkdir();  // Create the directory if it doesn't already exist
            }

            File file = new File(directory, fileName);
            ImageIO.write(bufferedImage, "PNG", file);
            System.out.println("Chart saved as " + fileName);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }
}
