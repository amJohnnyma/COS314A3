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
        MLP mlp = new MLP("src/data/BTC_train.csv", 10, 1, 5, 0, 0.001);
        mlp.trainNetwork(100);

        //graphing
        Graph lg = new Graph(mlp.getLosses());
        lg.createAndShowChart();

      //  Graph gg = new Graph(mlp.getGradients());
       // gg.createAndShowChart();
    }
}
