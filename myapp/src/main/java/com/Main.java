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
        MLP mlp = new MLP(" ", 10, 5);
        mlp.solveProblem(1);
    }
}
