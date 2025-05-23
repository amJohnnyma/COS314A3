package com;

/**
 * Main class to run the Cryptocurrency Stock Predictor
 */
public class Main 
{
    public static void main( String[] args )
    {
        System.out.println("=== Cryptocurrency Stock Predictor ===\n");
        
        // Example 1: Use default random seed (different each run)
        System.out.println("Example 1: Default random seed");
        CryptoStockPredictor predictor1 = new CryptoStockPredictor();
        predictor1.DT();
        
        System.out.println("\n=== Test Complete ===\n");
        
        // Example 2: Generate new random seed for another run
        System.out.println("\nExample 2: New random seed for same instance");
        predictor1.randomizeSeed();
        predictor1.DT();
        
        System.out.println("\n=== Test Complete ===\n");
        
        // Example 3: Use specific seed for reproducible results
        System.out.println("\nExample 3: Fixed seed for reproducible results");
        CryptoStockPredictor predictor2 = new CryptoStockPredictor(42);
        predictor2.DT();
        
        System.out.println("\n=== Test Complete ===\n");
        
        // Example 4: Switch between random and fixed seed
        System.out.println("\nExample 4: Switch to random seed mode");
        predictor2.setUseRandomSeed(true); // This will generate a new random seed
        predictor2.DT();
        
        System.out.println("\n=== All Tests Complete ===");
        
        // Uncomment to test with unpruned tree
        // predictor1.setUnpruned(true);
        // predictor1.DT();
    }
}