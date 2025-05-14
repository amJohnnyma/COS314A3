package com;

/**
 * Hello world!
 *
 */
public class Main
{
    public static void main( String[] args )
    {
        System.out.println( "Hello World!" );
        MLP mlp = new MLP(" ", 10, 10, 0, 0.5);
        mlp.trainNetwork(2);
    }
}
