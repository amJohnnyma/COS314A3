package com;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.FileReader;


public class TrainingMetrics {
    public Vector<Double> losses;
    public Vector<Double> avgWeights;
    public Vector<Double> avgBiases;
    public Vector<Long> epochTimes; // in milliseconds
    public Vector<Double> deltaValues;

    public TrainingMetrics(Vector<Double> losses, Vector<Double> avgWeights, Vector<Double> avgBiases,
                           Vector<Long> epochTimes, Vector<Double> deltaValues) {
        this.losses = losses;
        this.avgWeights = avgWeights;
        this.avgBiases = avgBiases;
        this.epochTimes = epochTimes;
        this.deltaValues = deltaValues;
    }

public void saveRawData(String fileName) {
        try {
            java.io.File dir = new java.io.File("WithStops");
            if (!dir.exists()) {
                dir.mkdir();
            }
            try (FileWriter writer = new FileWriter(new java.io.File(dir, fileName))) {
                Gson gson = new GsonBuilder().setPrettyPrinting().create();
                gson.toJson(this, writer);
                System.out.println("Raw training data saved as " + fileName);
            }
        } catch (IOException e) {
            System.err.println("Failed to save raw data: " + e.getMessage());
        }
    }
public static TrainingMetrics loadFromFile(String filePath) {
    try {
        Gson gson = new Gson();
        FileReader reader = new FileReader(filePath);
        return gson.fromJson(reader, TrainingMetrics.class);
    } catch (IOException e) {
        System.err.println("Failed to load training data: " + e.getMessage());
        return null;
    }
}


public static Vector<Double> smooth(List<Double> values, int window) {
    Vector<Double> smoothed = new Vector<>();
    double sum = 0;
    int w = Math.min(window, values.size());
    for (int i = 0; i < values.size(); i++) {
        sum += values.get(i);
        if (i >= w) sum -= values.get(i - w);
        if (i >= w - 1) smoothed.add(sum / w);
        else smoothed.add(sum / (i + 1));
    }
    return smoothed;
}

public static Vector<Double> smoothEpochTimes(Vector<Long> times, int window) {
    if (times == null) return null;
    return smooth(times.stream().map(t -> t / 1000.0).toList(), window);
}

}
