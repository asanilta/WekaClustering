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
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;

/**
 *
 * @author ASUS X202E
 */
public class AgnesMain {
    public static void main(String[] args) {
//        Instances data = loadData("C:\\Program Files\\Weka-3-8\\data\\weather.numeric.arff");
        Instances data = loadData("src/Dataset/weather.numeric.arff");
        MyAgnes agnes = new MyAgnes("single",data);
        agnes.printClusters();
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
