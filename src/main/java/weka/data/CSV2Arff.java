package weka.data;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

/**
 * Created by equipo on 17/11/2017.
 */
public class CSV2Arff {

    public static void main(String[] args) throws Exception {

        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("articulo/data/real/condiciones_vida/condiciones_vida_2016_training_updated.csv"));
        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("articulo/data/real/condiciones_vida/condiciones_vida_2016_training_updated.arff"));
        //saver.setDestination(new File("data/parkinson/s6/d897_female_motor_nms55_s6.arff"));
        saver.writeBatch();
    }
}
