/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kmeans;

import java.util.ArrayList;
import java.util.HashMap;
import weka.core.Attribute;
import weka.core.Instance;
import java.util.Scanner;
import weka.core.Instances;
import weka.core.EuclideanDistance;
/**
 *
 * @author ASUS X202E
 */
public class KMeansMain {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        System.out.print("Put number cluster : ");
       Scanner scanner = new Scanner(System.in);
        int numCluster = scanner.nextInt();
        MyKMeans km = new MyKMeans("src/Dataset/weather.arff",numCluster);
        km.doKMeans();
        
    }
}
