package com;

/**
 * Hello world!
 *
 */
public class Main
{
    public static void main( String[] args )
    {
        //HS: 16-32 HL: 1-2 LR: 0.0001 - 0.00001
        MLP mlp = new MLP("src/data/BTC_train.csv", 16, 1, 5, 0, 0.0005); 
        mlp.trainNetwork(100);

        //graphing
        Graph lg = new Graph(mlp.getLosses());
        lg.createAndShowChart();

      //  Graph gg = new Graph(mlp.getGradients());
       // gg.createAndShowChart();
    }
}
