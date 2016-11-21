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
        MyKMeans km = new MyKMeans("src/Dataset/weather.numeric.arff",5);
        Instances data = km.dataSource ;
        Attribute attr = data.attribute(0);
        System.out.println(attr.isNominal()+"|"+data.get(2).value(attr));
        
    }
}
