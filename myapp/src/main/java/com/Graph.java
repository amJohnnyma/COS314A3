package com;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Vector;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

public class Graph {

    private List<Double> losses;
    private List<Double> avgWeights;
    private List<Double> avgBiases;
    private List<Double> epochTimes;
    private List<Double> deltaValues;

    public Graph(List<Double> losses, List<Double> avgWeights, List<Double> avgBiases,
                List<Double> epochTimesInSec, List<Double> deltaValues) {

        this.losses = losses;
        this.avgWeights = avgWeights;
        this.avgBiases = avgBiases;
        this.epochTimes = epochTimesInSec;
        this.deltaValues = deltaValues;
    }

    public void createChart(String name) {
        // First save the data to CSV for reference
        saveDataToCsv(name.replace(".png", ".csv"));
        
        // Then create a simple chart image
        createSimpleChartImage(name);
    }
    
    private void saveDataToCsv(String fileName) {
        try {
            // Create a directory called "SmoothedGraphs" if it doesn't exist
            File directory = new File("SmoothedGraphs");
            if (!directory.exists()) {
                directory.mkdir();
            }
            
            File file = new File(directory, fileName);
            
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
                // Write CSV header
                writer.write("Epoch,Loss");
                if (avgWeights != null && !avgWeights.isEmpty()) writer.write(",Avg Weights");
                if (avgBiases != null && !avgBiases.isEmpty()) writer.write(",Avg Biases");
                if (epochTimes != null && !epochTimes.isEmpty()) writer.write(",Epoch Time (s)");
                if (deltaValues != null && !deltaValues.isEmpty()) writer.write(",Delta");
                writer.newLine();
                
                // Write data rows
                for (int i = 0; i < losses.size(); i++) {
                    writer.write(String.valueOf(i + 1) + "," + losses.get(i));
                    
                    if (avgWeights != null && !avgWeights.isEmpty()) {
                        writer.write(",");
                        if (i < avgWeights.size()) writer.write(String.valueOf(avgWeights.get(i)));
                    }
                    
                    if (avgBiases != null && !avgBiases.isEmpty()) {
                        writer.write(",");
                        if (i < avgBiases.size()) writer.write(String.valueOf(avgBiases.get(i)));
                    }
                    
                    if (epochTimes != null && !epochTimes.isEmpty()) {
                        writer.write(",");
                        if (i < epochTimes.size()) writer.write(String.valueOf(epochTimes.get(i)));
                    }
                    
                    if (deltaValues != null && !deltaValues.isEmpty()) {
                        writer.write(",");
                        if (i < deltaValues.size()) writer.write(String.valueOf(deltaValues.get(i)));
                    }
                    
                    writer.newLine();
                }
            }
            System.out.println("Data saved to CSV: " + fileName);
        } catch (IOException e) {
            System.err.println("Error saving CSV: " + e.getMessage());
        }
    }
    
    private void createSimpleChartImage(String fileName) {
        // Define image dimensions
        int width = 800;
        int height = 600;
        int padding = 50; // Padding for labels and axes
        
        // Create a BufferedImage
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        
        // Set rendering hints for better quality
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        
        // Fill background
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, width, height);
        
        // Draw title
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 20));
        String title = "Chart: " + fileName.replace(".png", "");
        int titleWidth = g2d.getFontMetrics().stringWidth(title);
        g2d.drawString(title, (width - titleWidth) / 2, 30);
        
        // Calculate plot area
        int plotX = padding;
        int plotY = padding + 30; // Additional space for title
        int plotWidth = width - 2 * padding;
        int plotHeight = height - 2 * padding - 60; // Additional space for legend
        
        // Find the min and max values for scaling
        double maxValue = Double.NEGATIVE_INFINITY;
        double minValue = Double.POSITIVE_INFINITY;
        
        for (Double value : losses) {
            maxValue = Math.max(maxValue, value);
            minValue = Math.min(minValue, value);
        }
        
        // Add some margin to the max and min
        double range = maxValue - minValue;
        maxValue += range * 0.1;
        minValue -= range * 0.1;
        
        // Draw axes
        g2d.setColor(Color.BLACK);
        g2d.drawLine(plotX, plotY, plotX, plotY + plotHeight); // Y-axis
        g2d.drawLine(plotX, plotY + plotHeight, plotX + plotWidth, plotY + plotHeight); // X-axis
        
        // Draw axis labels
        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        g2d.drawString("Epoch", plotX + plotWidth / 2, plotY + plotHeight + 25);
        g2d.drawString("Value", plotX - 40, plotY + plotHeight / 2);
        
        // Draw Y-axis tick marks and labels
        int numYTicks = 5;
        for (int i = 0; i <= numYTicks; i++) {
            int y = plotY + plotHeight - (i * plotHeight / numYTicks);
            double value = minValue + (i * (maxValue - minValue) / numYTicks);
            
            g2d.drawLine(plotX - 5, y, plotX, y); // Tick mark
            g2d.drawString(String.format("%.2f", value), plotX - 45, y + 5);
        }
        
        // Plot the loss data (red line)
        if (!losses.isEmpty()) {
            g2d.setColor(Color.RED);
            g2d.setStroke(new BasicStroke(2));
            
            for (int i = 1; i < losses.size(); i++) {
                int x1 = plotX + ((i - 1) * plotWidth / losses.size());
                int y1 = plotY + plotHeight - (int)((losses.get(i - 1) - minValue) * plotHeight / (maxValue - minValue));
                
                int x2 = plotX + (i * plotWidth / losses.size());
                int y2 = plotY + plotHeight - (int)((losses.get(i) - minValue) * plotHeight / (maxValue - minValue));
                
                g2d.drawLine(x1, y1, x2, y2);
            }
            
            // Add a legend entry for Loss
            g2d.setFont(new Font("Arial", Font.PLAIN, 12));
            g2d.setColor(Color.RED);
            g2d.fillRect(plotX, height - 30, 15, 15);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Loss", plotX + 20, height - 18);
        }
        
        // Draw legend for CSV file
        g2d.setFont(new Font("Arial", Font.ITALIC, 12));
        g2d.setColor(Color.BLUE);
        String csvNote = "Full data saved to CSV file with the same name";
        g2d.drawString(csvNote, width - 300, height - 18);
        
        // Dispose the graphics context
        g2d.dispose();
        
        // Save the image
        try {
            File directory = new File("SmoothedGraphs");
            if (!directory.exists()) {
                directory.mkdir();
            }
            File file = new File(directory, fileName);
            ImageIO.write(image, "PNG", file);
            System.out.println("Chart saved as " + fileName);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }
}