package experiment;

import voltric.clustering.util.GenerateCompleteData;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.data.arff.ArffFileReader;
import voltric.model.DiscreteBayesNet;
import weka.bif.XmlBifReader;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 13/05/2018.
 *
 * Completa los datos reales
 */
public class LearnPartitionBN {

    public static void main(String[] args) throws Exception {
        /** Cargamos los datos reales con los que se aprende el MPM */
        DiscreteData oldData = DataFileLoader.loadDiscreteData("experiments/real/data/cond_12.arff");

        /** Cargamos el MPM*/
        List<String> latentVars = new ArrayList<>();
        latentVars.add("variable26");
        latentVars.add("variable14");
        latentVars.add("variable12");
        latentVars.add("variable24");

        // Cargamos el modelo del cual vamos a calcular su matriz de distancias de Hellinger y su distancia media por particion
        DiscreteBayesNet olhc_model = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("EAST_real_condvida_2.xml"), latentVars);
        /** Generamos el nuevo dataSet con los datos completados */
        DiscreteData completeData = GenerateCompleteData.generateMultidimensional(oldData, olhc_model);
        /** Exportamos los datos en formato ARFF*/
        DataFileLoader.saveDiscreteData(completeData, "EAST_cond_12_completed.arff");

        /** TODO: Despues de esto podriamos aprender la red a partir de los datos aqui o en bnLearn */
    }
}
