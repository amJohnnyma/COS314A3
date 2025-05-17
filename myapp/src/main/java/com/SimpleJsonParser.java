package com;

import java.io.*;
import java.util.*;

/**
 * A simple JSON parser to replace Gson dependency
 */
public class SimpleJsonParser {
    
    /**
     * Saves training metrics to a JSON file
     */
    public static void saveToJson(TrainingMetrics metrics, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write("{\n");
            
            // Write losses
            writer.write("  \"losses\": [\n");
            writeDoubleVector(writer, metrics.losses);
            writer.write("  ],\n");
            
            // Write avgWeights
            writer.write("  \"avgWeights\": [\n");
            writeDoubleVector(writer, metrics.avgWeights);
            writer.write("  ],\n");
            
            // Write avgBiases
            writer.write("  \"avgBiases\": [\n");
            writeDoubleVector(writer, metrics.avgBiases);
            writer.write("  ],\n");
            
            // Write epochTimes
            writer.write("  \"epochTimes\": [\n");
            writeLongVector(writer, metrics.epochTimes);
            writer.write("  ],\n");
            
            // Write deltaValues
            writer.write("  \"deltaValues\": [\n");
            writeDoubleVector(writer, metrics.deltaValues);
            writer.write("  ]\n");
            
            writer.write("}");
        }
    }
    
    private static void writeDoubleVector(BufferedWriter writer, Vector<Double> values) throws IOException {
        if (values == null || values.isEmpty()) {
            writer.write("  ");
            return;
        }
        
        for (int i = 0; i < values.size(); i++) {
            writer.write("    " + values.get(i));
            if (i < values.size() - 1) {
                writer.write(",\n");
            } else {
                writer.write("\n");
            }
        }
    }
    
    private static void writeLongVector(BufferedWriter writer, Vector<Long> values) throws IOException {
        if (values == null || values.isEmpty()) {
            writer.write("  ");
            return;
        }
        
        for (int i = 0; i < values.size(); i++) {
            writer.write("    " + values.get(i));
            if (i < values.size() - 1) {
                writer.write(",\n");
            } else {
                writer.write("\n");
            }
        }
    }
    
    /**
     * Loads training metrics from a JSON file
     */
    public static TrainingMetrics loadFromJson(String filePath) throws IOException {
        Vector<Double> losses = new Vector<>();
        Vector<Double> avgWeights = new Vector<>();
        Vector<Double> avgBiases = new Vector<>();
        Vector<Long> epochTimes = new Vector<>();
        Vector<Double> deltaValues = new Vector<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            String currentArray = null;
            
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                
                if (line.contains("\"losses\":")) {
                    currentArray = "losses";
                } else if (line.contains("\"avgWeights\":")) {
                    currentArray = "avgWeights";
                } else if (line.contains("\"avgBiases\":")) {
                    currentArray = "avgBiases";
                } else if (line.contains("\"epochTimes\":")) {
                    currentArray = "epochTimes";
                } else if (line.contains("\"deltaValues\":")) {
                    currentArray = "deltaValues";
                } else if (line.equals("],") || line.equals("]")) {
                    currentArray = null;
                } else if (currentArray != null && !line.equals("[")) {
                    // Remove trailing comma if present
                    if (line.endsWith(",")) {
                        line = line.substring(0, line.length() - 1);
                    }
                    
                    try {
                        if (currentArray.equals("losses")) {
                            losses.add(Double.parseDouble(line));
                        } else if (currentArray.equals("avgWeights")) {
                            avgWeights.add(Double.parseDouble(line));
                        } else if (currentArray.equals("avgBiases")) {
                            avgBiases.add(Double.parseDouble(line));
                        } else if (currentArray.equals("epochTimes")) {
                            epochTimes.add(Long.parseLong(line));
                        } else if (currentArray.equals("deltaValues")) {
                            deltaValues.add(Double.parseDouble(line));
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("Error parsing number from line: " + line);
                    }
                }
            }
        }
        
        return new TrainingMetrics(losses, avgWeights, avgBiases, epochTimes, deltaValues);
    }
}