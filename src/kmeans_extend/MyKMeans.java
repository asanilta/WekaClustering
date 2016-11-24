/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kmeans_extend;
import kmeans.*;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.EuclideanDistance;
import weka.core.Attribute;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.AbstractClusterer;

/**
 *
 * @author ASUS X202E
 */
public class MyKMeans  extends AbstractClusterer{
    
    Instances dataSource ;
    int noOfClusters; 
    int numberIteration ;
    Instances centroid ;
    Instances firstCentroid ;
    HashMap<Integer,Integer> clusteredInstance ; // HashMap antara nomor instance dan nomor cluster
    List<List<Integer>> listClusteredInstance ; //HashMap antara nomor cluster dan kumpulan nomor instance di cluster tersebut
    boolean finish ;
    
    MyKMeans(int numCluster) {
        numberIteration = 0;
        this.clusteredInstance = new HashMap<Integer,Integer>();
        this.noOfClusters = numCluster ;
        finish = false ;
    }
    
    void doKMeans() {
        System.out.println("\n======== KMeans Result ========");
        int numIteration = 0;
        System.out.println("Data : "+dataSource.relationName());
        System.out.println("Number cluster : "+this.noOfClusters);
        initiateClusteredInstance();
        //1. Pilih random centroid
        chooseFirstCentroid() ;
//        printFirstCentroid();
        //2. Hitung jarak tiap data dengan tiap centroid dan lakukan clustering
        clusteringInstance() ;
      
//        //3. Update Centroid 
//        // Clustering dan update centroid terus dilakukan sampai tidak ada hasil cluster yang berubah dari hasil cluster sebelumnya
        do {
            numIteration++;
            updateCentroid() ;
            clusteringInstance() ;
        } while (!finish);
        System.out.println("Total iteration : "+numIteration);
        printFirstCentroid();
        printFinalCentroid();
        printSummary();      
        printClusteringResultWithData() ;
        printListClusteredInstance();

    }
    
    void printFirstCentroid() {
        System.out.println("\n ==== First Centroid (chosen randomly) ====");
        for (int i=0;i<noOfClusters;i++) System.out.println("Centroid  "+i+" : "+firstCentroid.get(i));   
    }
    
    void printFinalCentroid() {
         System.out.println("\n ====*** Final Centroid ***====");
        for (int i=0;i<noOfClusters;i++) System.out.println("Centroid  "+i+" : "+centroid.get(i));        
    }
    
    void printClusteringResultWithData() {
        System.out.println("\n --- LIST DATA WITH NUMBER CLUSTER ---");
        for (int i=0;i<dataSource.size();i++) {
            System.out.println(dataSource.get(i)+" : "+clusteredInstance.get(i));
        }
    }
        
    public int numberOfClusters() {
        return noOfClusters;
    }
    
    void printSummary() {
        System.out.println("\n ------ SUMMARY ------");
        for (int i=0;i<noOfClusters;i++) {
            int total = listClusteredInstance.get(i).size();
            System.out.println("Total instance clustered in cluster  "+i+" : "+total+" ( "+(double)total*100/dataSource.size()+"% ) ");
        }        
    }
    
    void printListClusteredInstance() {
        System.out.println("\n --- LIST CLUSTER WITH  DATA ---");
        for (int i=0;i<noOfClusters;i++) {
            List<Integer> listInst = listClusteredInstance.get(i);
            System.out.println("Centroid "+i+" : ");
            for (int j=0;j<listInst.size();j++) {
                System.out.println("    "+dataSource.get(listInst.get(j)));
            }
        }
    }
    
    void updateCentroid() {
        for (int i=0;i<noOfClusters;i++) {
            //Update centroid per cluster
//            ArrayList<Integer> listNumInstance = listClusteredInst.get(i);
            for (int j=0;j<dataSource.numAttributes();j++) {
                //untuk seluruh atribut
               if (dataSource.attribute(j).isNominal()) {
                    updateCentroidForNominal(i,j);
                } else if (dataSource.attribute(j).isNumeric()) {
                    updateCentroidForNumeric(i,j);
                }
            }
        }
    }
    
    void updateCentroidForNumeric(int numCentroid, int numAttr) {
      //  System.out.println("Update centroid "+numCentroid+" attr "+dataSource.attribute(numAttr)+"|"+numAttr);
        List<Integer> listInst = listClusteredInstance.get(numCentroid);
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
       // System.out.println("Update centroid "+numCentroid+" attr "+dataSource.attribute(numAttr)+"|"+numAttr);
        int distinctValue = dataSource.attribute(numAttr).numValues();
       int[] countInst = new int[distinctValue] ;
       for (int i=0;i<distinctValue;i++) countInst[i]++;
       Attribute attr = dataSource.attribute(numAttr);
       List<Integer> listInst = listClusteredInstance.get(numCentroid);
       //Mencari nilai attribut paling banyak dalam 1 cluster
       for (int i=0 ; i<listInst.size();i++) {
           Instance inst = dataSource.get(listInst.get(i));
           if (!inst.isMissing(attr)) {
            String attrValue = inst.toString(attr);
            int indexValue = attr.indexOfValue(attrValue);
           // System.out.println(inst+"|"+attrValue+"|"+indexValue);
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
        ArrayList<Integer> tempList = new ArrayList<Integer>() ;
        Instances tempInst = new Instances(dataSource,0);
        EuclideanDistance ed = new EuclideanDistance(centroid);
        int[] pointList = new int[noOfClusters];
        int checkNumberChange = 0 ;
        for (int i=0;i<noOfClusters;i++) {
            pointList[i] = i ;
        } 
        for (int i=0;i<dataSource.numInstances();i++) {
            int clusterNumber = -1;
            Instance currentInst = dataSource.get(i);
            try {
                clusterNumber = ed.closestPoint(currentInst,centroid, pointList);
            }
            catch (Exception ex) {
                System.out.println("************** "+ex.toString());
            }
            int clusterBefore = clusteredInstance.put(i,clusterNumber);
            if (clusterNumber!=clusterBefore) checkNumberChange++;           
        }
        if (checkNumberChange!=0) finish=false ;
        else finish = true ;
        updateListClusteredInstance();
      //  printListClusteredInstance();
    }
    
    void updateListClusteredInstance() {
        listClusteredInstance = new ArrayList<List<Integer>>();
        for (int i=0;i<noOfClusters;i++) {
            List<Integer> temp = new ArrayList<Integer>() ;
            listClusteredInstance.add(temp);
        }
        for (int i=0;i<dataSource.size();i++) {
            int clustNum = clusteredInstance.get(i);
            List<Integer> temp = listClusteredInstance.get(clustNum);            
           temp.add(i);
            listClusteredInstance.set(clustNum, temp);                    
        }
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
        for (int i=1;i<=noOfClusters;i++) {
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
    
    public void buildClusterer(Instances data) {
        this.dataSource = data ;
        this.centroid = new Instances(dataSource,noOfClusters);
        this.firstCentroid = new Instances(dataSource,noOfClusters);
        initiateClusteredInstance();
        chooseFirstCentroid() ;
        clusteringInstance() ;
        do {
            numberIteration++;
            updateCentroid() ;
            clusteringInstance() ;
        } while (!finish);
    }
    
       public int clusterInstance(Instance instance) {
        int clusterNo = -1;
        EuclideanDistance ed = new EuclideanDistance(centroid);
        int[] pointList = new int[noOfClusters];
        for (int i=0;i<noOfClusters;i++) {
            pointList[i] = i ;
        }
        try {
            clusterNo = ed.closestPoint(instance,centroid, pointList);
        }
        catch (Exception ex) {
            System.out.println("************** "+ex.toString());
        }   
        return clusterNo;
    }
    
}
