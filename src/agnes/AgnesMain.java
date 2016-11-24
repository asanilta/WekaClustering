/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package agnes;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;

/**
 *
 * @author ASUS X202E
 */
public class AgnesMain {
    public static void main(String[] args) throws Exception {
//        Instances data = loadData("C:\\Program Files\\Weka-3-8\\data\\weather.numeric.arff");
        System.out.print("File: ");
        Scanner scanner = new Scanner(System.in);
        String filename = scanner.next();
        System.out.print("Number of clusters: ");
        int numCluster = scanner.nextInt();
        System.out.print("Single/complete: ");
        String link = scanner.next();
        Instances data = loadData("src/Dataset/"+filename);
        MyAgnes agnes = new MyAgnes(link,numCluster);
        agnes.buildClusterer(data);
        System.out.println("Cluster Hierarchies:\n");
        agnes.printClustersID();
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(agnes);
        eval.evaluateClusterer(data);
        System.out.println("Cluster Evaluation:");
        System.out.println(eval.clusterResultsToString());
//        agnes.printClusters();
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
