/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kmeans;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.EuclideanDistance;
import weka.core.Attribute;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author ASUS X202E
 */
public class MyKMeans {
    
    Instances dataSource ;
    int numCluster; 
    Instances centroid ;
    Instances firstCentroid ;
    HashMap<Integer,Integer> clusteredInstance ; // HashMap antara nomor instance dan nomor cluster
    ArrayList<ArrayList<Integer>> listClusteredInst;
    boolean finish ;
    
    MyKMeans(String filePath, int numCluster) {
        this.clusteredInstance = new HashMap<Integer,Integer>();
        this.numCluster = numCluster ;
        dataSource = loadData(filePath);
        this.centroid = new Instances(dataSource,numCluster);
        this.firstCentroid = new Instances(dataSource,numCluster);
        finish = false ;
    }
    
    void doKMeans() {
        initiateClusteredInstance();
        //1. Pilih random centroid
        chooseFirstCentroid() ;
        //2. Hitung jarak tiap data dengan tiap centroid dan lakukan clustering
        clusteringInstance() ;
        //3. Update Centroid 
        // Clustering dan update centroid terus dilakukan sampai tidak ada hasil cluster yang berubah dari hasil cluster sebelumnya
        do {
            updateCentroid() ;
            clusteringInstance() ;
        } while (!finish);
        printFirstCentroid();
        printFinalCentroid();
        printSummary();      
    }
    
    void printFirstCentroid() {
        System.out.println(" ==== First Centroid (chosen randomly) ====");
        for (int i=0;i<numCluster;i++) System.out.println("Centroid  "+i+" : "+centroid.get(i));   
    }
    
    void printFinalCentroid() {
         System.out.println(" ====*** Final Centroid ***====");
        for (int i=0;i<numCluster;i++) System.out.println("Centroid  "+i+" : "+centroid.get(i));        
    }
    
    void printClusteringResultWithData() {
        for (int i=0;i<dataSource.size();i++) {
            System.out.println(dataSource.get(i)+" : "+clusteredInstance.get(i));
        }
    }
    
    void printSummary() {
        for (int i=0;i<numCluster;i++) {
            int total = listClusteredInst.get(i).size();
            System.out.println("Total instance clustered in cluster  "+i+" : "+total+" ( "+(double)total/dataSource.size()+" ) ");
        }        
    }
    void updateCentroid() {
        for (int i=0;i<numCluster;i++) {
            //Update centroid per cluster
            ArrayList<Integer> listNumInstance = listClusteredInst.get(i);
            for (int j=0;j<dataSource.numAttributes();j++) {
                //untuk seluruh atribut
                if (dataSource.attribute(i).isNominal()) {
                    updateCentroidForNominal(i,j);
                } else if (dataSource.attribute(i).isNumeric()) {
                    updateCentroidForNumeric(i,j);
                }
            }
        }
    }
    
    void updateCentroidForNumeric(int numCentroid, int numAttr) {
        ArrayList<Integer> listInst = listClusteredInst.get(numCentroid);
        Attribute attr= dataSource.attribute(numAttr);
        double sum = 0 ;
        for (int i=0; i<listInst.size();i++) {
            Instance inst = dataSource.get(listInst.get(i));
            sum+= inst.value(attr);
        }
        double newValue = (double) sum / listInst.size();
        Instance tempCentroid = centroid.get(numCentroid);
        tempCentroid.setValue(attr, newValue);
        centroid.set(numCentroid, tempCentroid);
    }
    
    void updateCentroidForNominal(int numCentroid, int numAttr) {
       int distinctValue = dataSource.attribute(numAttr).numValues();
       int[] countInst = new int[distinctValue] ;
       for (int i=0;i<distinctValue;i++) countInst[i]++;
       Attribute attr = dataSource.attribute(numAttr);
       ArrayList<Integer> listInst = listClusteredInst.get(numCentroid);
       //Mencari nilai attribut paling banyak dalam 1 cluster
       for (int i=0 ; i<listInst.size();i++) {
           Instance inst = dataSource.get(listInst.get(i));
           if (!inst.isMissing(attr)) {
            String attrValue = inst.toString(attr);
            int indexValue = attr.indexOfValue(attrValue);
            countInst[indexValue]++;
           }
       }
       int max=-1 ,idxMax=-1;
       for (int i=0;i<distinctValue;i++) {
           if (countInst[i]>max) {
               idxMax=i;
               max = countInst[i];
           }
       }
       String newValue = attr.value(idxMax);
       Instance tempCentroid = centroid.get(numCentroid);
       tempCentroid.setValue(attr, newValue);
       centroid.set(numCentroid, tempCentroid);
    }
    
    
    //Prosedur ini akan mengkluster setiap instance dan mengecek apakah ada hasil clustering yang berubah dari sebelumnya
    //Jika hasil clustering sama dengan cluster sebelumnya, variabel "finish" akan bernilai true
    //Jika ada hasil cluster yang berubah, variabel "finish" bernilai false
    void clusteringInstance() {
        listClusteredInst = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> tempList = new ArrayList<Integer>() ;
        Instances tempInst = new Instances(dataSource,0);
        EuclideanDistance ed = new EuclideanDistance(centroid);
        int[] pointList = new int[numCluster];
        int checkNumberChange = 0 ;
        for (int i=0;i<numCluster;i++) {
            pointList[i] = i ;
            listClusteredInst.add(tempList);
        }
        for (int i=0;i<dataSource.numInstances();i++) {
            Instance currentInst = dataSource.get(i);
            try {
                int clusterNumber = ed.closestPoint(currentInst,centroid, pointList);
                int clusterBefore = clusteredInstance.put(i,clusterNumber);
                if (clusterNumber!=clusterBefore) checkNumberChange++;
                tempList = listClusteredInst.get(clusterNumber);
                tempList.add(i);
                listClusteredInst.set(clusterNumber, tempList);
            } catch (Exception ex) {
                System.out.println("************** "+ex.toString());
            }
        }
        if (checkNumberChange!=0) finish=false ;
        else finish = true ;
    }
    
    void initiateClusteredInstance() {
        for (int i=0;i<dataSource.numInstances();i++) {
            clusteredInstance.put(i, 0);
        }
    }
    
    void chooseFirstCentroid() {
        int totalData = dataSource.numInstances();
        ArrayList<Integer> tempNumber = new ArrayList<Integer>() ;
        Random random = new Random() ;
        int numberData ;
        for (int i=1;i<=numCluster;i++) {
            do {
                 numberData = random.nextInt(totalData);
            } while (tempNumber.contains(numberData)) ;
            tempNumber.add(numberData);
            firstCentroid.add(dataSource.get(numberData));
            centroid.add(dataSource.get(numberData));
        }
    }
    
    Instances loadData(String filePath) {
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
    
    void buildClusterer(Instances data) {
        
    }
    
}
