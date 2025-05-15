package com;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;


import java.util.*;
import java.util.List;

public class Graph {

    private Vector<Double> losses;
    private Vector<Double> avgWeights;
    private Vector<Double> avgBiases;
    private Vector<Long> epochTimes;
    private Vector<Double> deltaValues;


    public Graph(Vector<Double> losses, Vector<Double> avgWeights, Vector<Double> avgBiases, Vector<Long> epochTimes, Vector<Double> deltaValues) {
        this.losses = losses;
        this.avgWeights = avgWeights;
        this.avgBiases = avgBiases;
        this.epochTimes = epochTimes;
        this.deltaValues = deltaValues;
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
int smoothWindow = 10;

List<Double> smoothedLosses = movingAverage(losses, smoothWindow * 6);
List<Double> smoothedWeights = avgWeights != null ? movingAverage(avgWeights, smoothWindow) : null;
List<Double> smoothedBiases = avgBiases != null ? movingAverage(avgBiases, smoothWindow) : null;
List<Double> smoothedTimes = epochTimes != null ? movingAverage(epochTimes.stream().map(t -> t / 1000.0).toList(), smoothWindow) : null;
List<Double> smoothedDeltas = deltaValues != null ? movingAverage(deltaValues, smoothWindow * 6) : null;

for (int i = 0; i < smoothedLosses.size(); i++) {
    dataset.addValue(smoothedLosses.get(i), "Loss", Integer.toString(i + 1));
    if (smoothedWeights != null && i < smoothedWeights.size())
        dataset.addValue(smoothedWeights.get(i), "Avg Weights", Integer.toString(i + 1));
    if (smoothedBiases != null && i < smoothedBiases.size())
        dataset.addValue(smoothedBiases.get(i), "Avg Biases", Integer.toString(i + 1));
    if (smoothedTimes != null && i < smoothedTimes.size())
        dataset.addValue(smoothedTimes.get(i), "Epoch Time (s)", Integer.toString(i + 1));
    if (smoothedDeltas != null && i < smoothedDeltas.size())
        dataset.addValue(smoothedDeltas.get(i), "Delta", Integer.toString(i + 1));
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

        public List<Double> movingAverage(List<Double> values, int window) {
            List<Double> smoothed = new ArrayList<>();
            for (int i = 0; i < values.size(); i++) {
                int start = Math.max(0, i - window + 1);
                int end = i + 1;
                double sum = 0.0;
                for (int j = start; j < end; j++) {
                    sum += values.get(j);
                }
                smoothed.add(sum / (end - start));
            }
            return smoothed;
        }
}
