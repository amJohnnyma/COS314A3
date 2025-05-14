package com;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import javax.swing.*;
import java.util.Vector;

public class Graph {

    private Vector<Double> values;

    public Graph(Vector<Double> losses) {
        this.values = losses;
    }

    public void createAndShowChart() {
        // Create a dataset using the losses
        DefaultCategoryDataset dataset = createDataset();

        // Create a chart based on the dataset
        JFreeChart chart = ChartFactory.createLineChart(
                "Loss over Time",  // Chart title
                "Epoch",           // X-axis label
                "Loss",            // Y-axis label
                dataset,           // Dataset
                PlotOrientation.VERTICAL,  // Orientation (vertical)
                true,              // Include legend
                true,              // Tooltips
                false              // URLs
        );

        // Create a panel to display the chart
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));

        // Create a frame to display the chart
        JFrame frame = new JFrame("Loss Graph");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(chartPanel);
        frame.pack();
        frame.setVisible(true);
    }

    private DefaultCategoryDataset createDataset() {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        // Loop through the losses vector and add data to the dataset
        for (int i = 0; i < values.size(); i++) {
            dataset.addValue(values.get(i), "Loss", Integer.toString(i + 1));  // Use epoch index for x-axis
        }

        return dataset;
    }
}
