/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kmeans_extend;

import java.io.BufferedReader;
import java.io.FileReader;
import kmeans.*;
import java.util.ArrayList;
import java.util.HashMap;
import weka.core.Attribute;
import weka.core.Instance;
import java.util.Scanner;
import weka.clusterers.ClusterEvaluation;
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
        Instances data = loadData("src/Dataset/weather.arff");
        MyKMeans kmeans = new MyKMeans(numCluster);
        kmeans.buildClusterer(data);
        kmeans.printFinalCentroid();
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(kmeans);
        eval.evaluateClusterer(data);
        System.out.println("\n==== Evaluation Result ====");
        System.out.println(eval.clusterResultsToString());
        
        
    }
    
        public static Instances loadData (String filePath) {
            BufferedReader reader;
            Instances data = null ;
            try {
                reader = new BufferedReader(new FileReader(filePath)); 
                data = new Instances(reader);
                reader.close();
                data.setClassIndex(data.numAttributes() - 1);
            } catch (Exception e) {
                
            }
            return data ;
        }

}
