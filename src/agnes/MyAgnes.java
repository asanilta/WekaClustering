/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package agnes;
import weka.clusterers.AbstractClusterer;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities;
import weka.core.Attribute;
import java.util.ArrayList;
import weka.core.Utils;
import weka.core.EuclideanDistance;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Objects;

/**
 *
 * @author ASUS X202E
 */
public class MyAgnes extends AbstractClusterer {
    String link;
    int noOfClusters;
    HashMap<Integer[],Double> distanceMatrix = new HashMap<>();
    ArrayList<ArrayList<ArrayList<Instance>>> finalClusters = new ArrayList<>();
    EuclideanDistance distanceCounter;
//    HashMap<Instance,Integer> instanceID = new HashMap<>();
    
    MyAgnes(String _link, int _noOfClusters) {
        link = _link;
        noOfClusters = _noOfClusters;
    }
    
    public void setLink(String _link) {
        link = _link;
    }
    
    public void setNoOfClusters(int _noOfClusters) {
        noOfClusters = _noOfClusters;
    }
    
    public int numberOfClusters() {
        return noOfClusters;
    }
    
    public void buildClusterer(Instances data) {
        distanceCounter = new EuclideanDistance(data);
        ArrayList<ArrayList<Instance>> currentClusters = new ArrayList<>();
        for (int i=0;i<data.numInstances();i++) {
            currentClusters.add(new ArrayList<>());
            currentClusters.get(i).add(data.instance(i));
//            instanceID.put(data.instance(i),i);
        }
        addNewClusterHierarchy(currentClusters);
    }
    
    private void addNewClusterHierarchy(ArrayList<ArrayList<Instance>> currentClusters) {
        finalClusters.add(currentClusters);
//        for (int i=0;i<currentClusters.size();i++) printCluster(currentClusters.get(i));
        if (currentClusters.size()>noOfClusters) {
            updateDistanceMatrix(currentClusters);
            Integer[] key = findClosestClusters(currentClusters);
//            System.out.println("Closest clusters: "+key[0]+" "+key[1]);
            ArrayList<ArrayList<Instance>> newClusters = new ArrayList<>();
            for (int j=0;j<currentClusters.size();j++) {
                newClusters.add(new ArrayList<>());
                newClusters.get(j).addAll(currentClusters.get(j));
            }
            newClusters.get(key[0]).addAll(newClusters.get(key[1]));
            newClusters.remove((int)key[1]);
            addNewClusterHierarchy(newClusters);
        }
    }
   
    private Integer[] findClosestClusters(ArrayList<ArrayList<Instance>> clusters) {
        Double minValue = Collections.min(distanceMatrix.values());
        for (Entry<Integer[],Double> entry : distanceMatrix.entrySet()) {
            if (Objects.equals(entry.getValue(), minValue)) return entry.getKey();
        }
        return null;
    }
    
    private void updateDistanceMatrix(ArrayList<ArrayList<Instance>> clusters) {
        distanceMatrix.clear();
        for (int i=0;i<clusters.size()-1;i++) {
            for (int j=i+1;j<clusters.size();j++) {        
               Integer[] array = new Integer[2];
                array[0] = i;
                array[1] = j;
                if (link.equals("single")) {
                    distanceMatrix.put(array,findClosestDistance(clusters.get(i),clusters.get(j)));
                } else if (link.equals("complete")) {
                    distanceMatrix.put(array,findFurthestDistance(clusters.get(i),clusters.get(j)));
                }
            }
        }
    }
    
    private double findClosestDistance(ArrayList<Instance> cluster1, ArrayList<Instance> cluster2) {
        ArrayList<Double> distances = new ArrayList<>();
        for (Instance instance1 : cluster1) {
            for (Instance instance2 : cluster2) {
                distances.add(distanceCounter.distance(instance1,instance2));
            }
        }
        return Collections.min(distances);
    }
    
    private double findFurthestDistance(ArrayList<Instance> cluster1, ArrayList<Instance> cluster2) {
        ArrayList<Double> distances = new ArrayList<>();
        for (Instance instance1 : cluster1) {
            for (Instance instance2 : cluster2) {
                distances.add(distanceCounter.distance(instance1,instance2));
            }
        }
        return Collections.max(distances);
    }
    
    public void printClusters() {
        for (int i=0;i<finalClusters.size();i++) {
            ArrayList<ArrayList<Instance>> clusterHierarchy = finalClusters.get(i);
            System.out.println(clusterHierarchy.size()+" CLUSTERS:\n");
            int j;
            for (j=0;j<clusterHierarchy.size();j++) {
                System.out.println("Cluster "+j);
                printCluster(clusterHierarchy.get(j));
                System.out.println("");
            }
            System.out.println("-----------------\n");
            /*
            ArrayList<ArrayList<Instance>> clusterHierarchy = finalClusters.get(i);
            System.out.println(clusterHierarchy.size()+" Clusters:");
            int j;
            for (j=0;j<clusterHierarchy.size()-1;j++) {
                printCluster(clusterHierarchy.get(j));
                System.out.print(",");
            }
            printCluster(clusterHierarchy.get(j));
            System.out.println("\n");*/
        }
    }
    
    private void printCluster(ArrayList<Instance> cluster) {
        /*System.out.print("(");
        int i;
        for (i=0;i<cluster.size()-1;i++) {
            System.out.println(cluster.get(i).toString());
        }
        System.out.print(instanceID.get(cluster.get(i))+")");*/
        for (int i=0;i<cluster.size();i++) {
            System.out.println(cluster.get(i).toString());
        }
    }
}
