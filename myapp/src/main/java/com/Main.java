package com;

/**
 * Hello world!
 *
 */
public class Main
{
    public static void main( String[] args )
    {
        //System.out.println( "Hello World!" );
        MLP mlp = new MLP("src/data/BTC_train.csv", 10, 5, 0, 0.01);
        mlp.trainNetwork(2);
    }
}
