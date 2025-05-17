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
import java.awt.Stroke;

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
        
        // Then create an enhanced chart image
        createEnhancedChartImage(name);
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
    
    private void createEnhancedChartImage(String fileName) {
        // Define image dimensions
        int width = 800;
        int height = 600;
        int padding = 60; // Padding for labels and axes
        
        // Create a BufferedImage
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        
        // Set rendering hints for better quality
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
        
        // Define colors
        Color backgroundColor = Color.WHITE;
        Color titleColor = Color.BLACK;
        Color gridColor = new Color(220, 220, 220);
        Color axisColor = Color.BLACK;
        Color lossColor = Color.RED;
        Color weightColor = Color.BLUE;
        Color biasColor = Color.MAGENTA;
        Color deltaColor = Color.GREEN;
        Color timeColor = Color.ORANGE;
        
        // Fill background
        g2d.setColor(backgroundColor);
        g2d.fillRect(0, 0, width, height);
        
        // Draw title
        g2d.setColor(titleColor);
        g2d.setFont(new Font("Arial", Font.BOLD, 16));
        String title = fileName.replace(".png", "");
        int titleWidth = g2d.getFontMetrics().stringWidth(title);
        g2d.drawString(title, (width - titleWidth) / 2, 25);
        
        // Calculate plot area
        int plotX = padding;
        int plotY = padding;
        int plotWidth = width - 2 * padding;
        int plotHeight = height - 2 * padding - 40; // Additional space for legend
        
        // Draw plot background
        g2d.setColor(new Color(240, 240, 240));
        g2d.fillRect(plotX, plotY, plotWidth, plotHeight);
        
        // Find the min and max values for scaling across all datasets
        double maxValue = Double.NEGATIVE_INFINITY;
        double minValue = Double.POSITIVE_INFINITY;
        
        // Check all datasets for min/max
        if (losses != null && !losses.isEmpty()) {
            for (Double value : losses) {
                if (value != null) {
                    maxValue = Math.max(maxValue, value);
                    minValue = Math.min(minValue, value);
                }
            }
        }
        
        if (avgWeights != null && !avgWeights.isEmpty()) {
            for (Double value : avgWeights) {
                if (value != null) {
                    maxValue = Math.max(maxValue, value);
                    minValue = Math.min(minValue, value);
                }
            }
        }
        
        if (avgBiases != null && !avgBiases.isEmpty()) {
            for (Double value : avgBiases) {
                if (value != null) {
                    maxValue = Math.max(maxValue, value);
                    minValue = Math.min(minValue, value);
                }
            }
        }
        
        if (deltaValues != null && !deltaValues.isEmpty()) {
            for (Double value : deltaValues) {
                if (value != null) {
                    maxValue = Math.max(maxValue, value);
                    minValue = Math.min(minValue, value);
                }
            }
        }
        
        // Epoch times are typically on a different scale, so don't include in auto-scaling
        
        // If no valid data was found, set default range
        if (maxValue == Double.NEGATIVE_INFINITY) {
            maxValue = 1.0;
            minValue = 0.0;
        }
        
        // Add some margin to the min/max and ensure min < max
        double range = maxValue - minValue;
        if (range == 0) range = 1.0; // Prevent division by zero
        
        maxValue += range * 0.1;
        minValue -= range * 0.1;
        
        // Ensure we include zero in the range if it's close
        if (minValue > 0 && minValue < range * 0.3) minValue = 0;
        if (maxValue < 0 && maxValue > -range * 0.3) maxValue = 0;
        
        // Draw grid lines
        g2d.setColor(gridColor);
        Stroke dashedStroke = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 
                                             10.0f, new float[] {4.0f}, 0.0f);
        g2d.setStroke(dashedStroke);
        
        // Horizontal grid lines
        int numYGrids = 10;
        for (int i = 0; i <= numYGrids; i++) {
            int y = plotY + plotHeight - (i * plotHeight / numYGrids);
            g2d.drawLine(plotX, y, plotX + plotWidth, y);
        }
        
        // Vertical grid lines
        int numXGrids = 8;
        for (int i = 0; i <= numXGrids; i++) {
            int x = plotX + (i * plotWidth / numXGrids);
            g2d.drawLine(x, plotY, x, plotY + plotHeight);
        }
        
        // Draw axes with solid stroke
        g2d.setColor(axisColor);
        g2d.setStroke(new BasicStroke(1.5f));
        g2d.drawLine(plotX, plotY, plotX, plotY + plotHeight); // Y-axis
        g2d.drawLine(plotX, plotY + plotHeight, plotX + plotWidth, plotY + plotHeight); // X-axis
        
        // Draw axis labels
        g2d.setFont(new Font("Arial", Font.BOLD, 14));
        g2d.drawString("Epoch", plotX + plotWidth / 2, height - 18);
        
        // Rotate the Y-axis label
        g2d.translate(15, plotY + plotHeight / 2);
        g2d.rotate(-Math.PI / 2);
        g2d.drawString("Value", 0, 0);
        g2d.rotate(Math.PI / 2);
        g2d.translate(-15, -(plotY + plotHeight / 2));
        
        // Draw Y-axis tick marks and labels
        g2d.setFont(new Font("Arial", Font.PLAIN, 10));
        int numYTicks = 10;
        for (int i = 0; i <= numYTicks; i++) {
            int y = plotY + plotHeight - (i * plotHeight / numYTicks);
            double value = minValue + (i * (maxValue - minValue) / numYTicks);
            
            g2d.setColor(axisColor);
            g2d.drawLine(plotX - 5, y, plotX, y); // Tick mark
            g2d.drawString(String.format("%.1f", value), plotX - 40, y + 4);
        }
        
        // Plot the data - using thicker lines
        BasicStroke thickStroke = new BasicStroke(2.0f);
        g2d.setStroke(thickStroke);
        
        int maxDataPoints = losses.size();
        
        // Plot Loss (Red)
        if (losses != null && !losses.isEmpty()) {
            g2d.setColor(lossColor);
            
            for (int i = 1; i < losses.size(); i++) {
                Double value1 = losses.get(i - 1);
                Double value2 = losses.get(i);
                
                if (value1 != null && value2 != null) {
                    int x1 = plotX + ((i - 1) * plotWidth / maxDataPoints);
                    int y1 = plotY + plotHeight - (int)((value1 - minValue) * plotHeight / (maxValue - minValue));
                    
                    int x2 = plotX + (i * plotWidth / maxDataPoints);
                    int y2 = plotY + plotHeight - (int)((value2 - minValue) * plotHeight / (maxValue - minValue));
                    
                    g2d.drawLine(x1, y1, x2, y2);
                }
            }
        }
        
        // Plot Average Weights (Blue)
        if (avgWeights != null && !avgWeights.isEmpty()) {
            g2d.setColor(weightColor);
            
            for (int i = 1; i < avgWeights.size() && i < maxDataPoints; i++) {
                Double value1 = avgWeights.get(i - 1);
                Double value2 = avgWeights.get(i);
                
                if (value1 != null && value2 != null) {
                    int x1 = plotX + ((i - 1) * plotWidth / maxDataPoints);
                    int y1 = plotY + plotHeight - (int)((value1 - minValue) * plotHeight / (maxValue - minValue));
                    
                    int x2 = plotX + (i * plotWidth / maxDataPoints);
                    int y2 = plotY + plotHeight - (int)((value2 - minValue) * plotHeight / (maxValue - minValue));
                    
                    g2d.drawLine(x1, y1, x2, y2);
                }
            }
        }
        
        // Plot Average Biases (Magenta)
        if (avgBiases != null && !avgBiases.isEmpty()) {
            g2d.setColor(biasColor);
            
            for (int i = 1; i < avgBiases.size() && i < maxDataPoints; i++) {
                Double value1 = avgBiases.get(i - 1);
                Double value2 = avgBiases.get(i);
                
                if (value1 != null && value2 != null) {
                    int x1 = plotX + ((i - 1) * plotWidth / maxDataPoints);
                    int y1 = plotY + plotHeight - (int)((value1 - minValue) * plotHeight / (maxValue - minValue));
                    
                    int x2 = plotX + (i * plotWidth / maxDataPoints);
                    int y2 = plotY + plotHeight - (int)((value2 - minValue) * plotHeight / (maxValue - minValue));
                    
                    g2d.drawLine(x1, y1, x2, y2);
                }
            }
        }
        
        // Plot Delta Values (Green)
        if (deltaValues != null && !deltaValues.isEmpty()) {
            g2d.setColor(deltaColor);
            
            for (int i = 1; i < deltaValues.size() && i < maxDataPoints; i++) {
                Double value1 = deltaValues.get(i - 1);
                Double value2 = deltaValues.get(i);
                
                if (value1 != null && value2 != null) {
                    int x1 = plotX + ((i - 1) * plotWidth / maxDataPoints);
                    int y1 = plotY + plotHeight - (int)((value1 - minValue) * plotHeight / (maxValue - minValue));
                    
                    int x2 = plotX + (i * plotWidth / maxDataPoints);
                    int y2 = plotY + plotHeight - (int)((value2 - minValue) * plotHeight / (maxValue - minValue));
                    
                    g2d.drawLine(x1, y1, x2, y2);
                }
            }
        }
        
        // Draw legend
        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        int legendX = plotX;
        int legendY = height - 25;
        int legendItemWidth = 120;
        
        if (losses != null && !losses.isEmpty()) {
            g2d.setColor(lossColor);
            g2d.fillRect(legendX, legendY, 16, 8);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Loss", legendX + 20, legendY + 8);
            legendX += legendItemWidth;
        }
        
        if (avgWeights != null && !avgWeights.isEmpty()) {
            g2d.setColor(weightColor);
            g2d.fillRect(legendX, legendY, 16, 8);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Avg Weights", legendX + 20, legendY + 8);
            legendX += legendItemWidth;
        }
        
        if (avgBiases != null && !avgBiases.isEmpty()) {
            g2d.setColor(biasColor);
            g2d.fillRect(legendX, legendY, 16, 8);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Avg Biases", legendX + 20, legendY + 8);
            legendX += legendItemWidth;
        }
        
        if (deltaValues != null && !deltaValues.isEmpty()) {
            g2d.setColor(deltaColor);
            g2d.fillRect(legendX, legendY, 16, 8);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Delta", legendX + 20, legendY + 8);
            legendX += legendItemWidth;
        }
        
        if (epochTimes != null && !epochTimes.isEmpty()) {
            g2d.setColor(timeColor);
            g2d.fillRect(legendX, legendY, 16, 8);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Epoch Time (s)", legendX + 20, legendY + 8);
        }
        
        // Draw note about CSV file
        g2d.setFont(new Font("Arial", Font.ITALIC, 10));
        g2d.setColor(Color.BLUE);
        String csvNote = "Full data saved to CSV file with the same name";
        int noteWidth = g2d.getFontMetrics().stringWidth(csvNote);
        g2d.drawString(csvNote, width - noteWidth - 10, height - 10);
        
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