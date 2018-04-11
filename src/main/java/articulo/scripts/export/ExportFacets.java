package articulo.scripts.export;

import articulo.MBCodigo;
import articulo.facet.DataFacet;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;
import weka.bif.XmlBifReader;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by equipo on 11/04/2018.
 */
public class ExportFacets {

    public static void main(String[] args) throws Exception {

        // Parkinson
        String d897 = "data/parkinson/s6/d897_motor_nms55_s6.arff";
        String d897_motor = "data/parkinson/motor/d897_motor25.arff";
        String d897_nms = "data/parkinson/s6/nms/d897_nms30_s6.arff";
        String d897_noise = "data/parkinson/s6/d897_noise.arff";

        // Beeps
        String beeps2013 = "articulo/data/real/beeps/beeps2013_sampled_training_updated.arff";

        // Condiciones_vida
        String condiciones_vida = "articulo/data/real/condiciones_vida/condiciones_vida_2016_training.arff";

        // MENA - Egypt 2013
        String egypt2013 = "articulo/data/real/mena-egypt/Egypt_1927_training.arff";

        String dataString = condiciones_vida;
        String dataName = "condiciones_vida";

        /** Load Weka data */
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(dataString));
        Instances dataWeka = loader.getDataSet();
        dataWeka.setClassIndex(0);

        /** Load Voltric data */
        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData(dataString);

        MBCodigo codigo = new MBCodigo();

        /** Facets generated from data (BN is learned) */
        //List<DataFacet> facets = codigo.facetDeterminationWithMarkovBlanket(dataVoltric, dataWeka);

        /**Facets generated from a loaded BN */
        DiscreteBayesNet loadedBn = XmlBifReader.processFile(new File("estudios/condiciones_vida/condiciones_vida_2016.xml"));
        List<DataFacet> facets = codigo.facetDeterminationWithMarkovBlanket(loadedBn);

        /** Export all Facets */
        exportAllFacets(facets, dataName + "_allMBs.txt", dataVoltric);

        /** Export non-repeated Facets */
        exportNonRepeatedFacets(facets, dataName + "_non-repeatedMBs.txt");
    }

    public static void exportAllFacets(List<DataFacet> facets, String filePath, DiscreteData dataVoltric) throws Exception {
        PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filePath), "UTF-8"));

        // Print the number of facets
        writer.println("Number of facets: " + facets.size());

        // Cada uno de los MBs se corresponde con una variable, el indice de la variable coincide con el indice del MB
        for(int i = 0; i < facets.size(); i++){

            DiscreteVariable mbMainVar = dataVoltric.getVariables().get(i);
            DiscreteBayesNet mb = facets.get(i).getBayesNet();
            // Print the 'main' variable of the MB
            writer.println("\n\n"+ mbMainVar.getName());

            // Print all the rest of variables
            for(DiscreteVariable mbVar: mb.getVariables())
                writer.print(mbVar.getName() + "; ");
        }
        writer.close();
    }

    // Export non-repeated facets with facets whose MB is not empty
    public static void exportNonRepeatedFacets(List<DataFacet> facets, String filePath) throws Exception {
        PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filePath), "UTF-8"));

        List<DataFacet> nonEmptyFacets = facets.stream().filter(x->x.getVariables().size() > 1).collect(Collectors.toList());

        Set<DataFacet> nonRepeatedDataFacets = new LinkedHashSet<>();

        for(DataFacet facet: nonEmptyFacets)
            nonRepeatedDataFacets.add(facet);

        // Print the number of facets
        writer.println("Number of facets: " + facets.size());
        writer.println("Number of nonrepeated facets: " + nonRepeatedDataFacets.size());

        // Iteramos por cada una de las facetas y las escribimos en el archivo
        for(DataFacet facet: nonRepeatedDataFacets){
            DiscreteBayesNet mb = facet.getBayesNet();

            // Print the MB's variables
            for(DiscreteVariable mbVar: mb.getVariables())
                writer.print(mbVar.getName() + "; ");

            writer.println("\n\n");
        }
        writer.close();
    }
}
