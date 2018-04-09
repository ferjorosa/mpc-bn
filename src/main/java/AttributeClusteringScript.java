import articulo.MBCodigo;
import articulo.facet.DataFacet;
import articulo.scripts.export.ExportMatrixCSV;
import util.HashBnManager;
import util.SimpleHashCreator;
import voltric.clustering.singleview.HiddenNaiveBayes;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.bif.OldBifFileWriter;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.util.distance.NID;
import voltric.variables.DiscreteVariable;
import weka.bif.XmlBifReader;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.*;

/**
 * - Exporta el LCM asociado a cada una de las facetas escogidas
 * - Exporta la matriz NID entre cada par de facetas escogidas
 * TODO: tiempos de ejecucion
 * TODO: Exportar MBs?
 */
public class AttributeClusteringScript {

    private static Map<Long, DiscreteBayesNet> loadedBns = new HashMap<>();
    private static MBCodigo codigo = new MBCodigo();

    public static void main(String[] args) throws Exception {

        // Parkinson
        String d897 = "data/parkinson/s6/d897_motor_nms55_s6.arff";
        String d897_motor = "data/parkinson/motor/d897_motor25.arff";
        String d897_nms = "data/parkinson/s6/nms/d897_nms30_s6.arff";
        String d897_noise = "data/parkinson/s6/d897_noise.arff";

        // Beeps
        String beeps2013 = "articulo/data/real/beeps/beeps2013_sampled_training_updated.arff";

        // Condiciones_vida
        String condiciones_vida = "estudios/condiciones_vida/data/condiciones_vida_2016_training.arff";

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

        double nbTimeStart = System.currentTimeMillis();

        /** Facets generated from a loaded BN */
        DiscreteBayesNet loadedBn = XmlBifReader.processFile(new File("estudios/condiciones_vida/condiciones_vida_2016.xml"));
        List<DataFacet> facets = codigo.facetDeterminationWithMarkovBlanket(loadedBn);
        List<DataFacet> nonRepeatedDataFacets = filterDataFacets(facets);

        /** LCMs genearated from the filtered list of facets */
        DiscreteParameterLearning em = new ParallelEM(new EmConfig(), ScoreType.BIC);
        for(DataFacet facet: nonRepeatedDataFacets){
            LearningResult<DiscreteBayesNet> lcmResult = HiddenNaiveBayes.learnModel(20, dataVoltric.project(facet.getVariables()), em, 1e-4);

            // Export model in BIF 0.15 format
            OutputStream nbOutput = new FileOutputStream("estudios/"+dataName+"/facets/lcm/"+facet.getVariables().get(0).getName()+".bif");
            BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
            writer.write(lcmResult.getBayesianNetwork());

            // Export model in OBIF format
            OldBifFileWriter.writeBif("estudios/"+dataName+"/facets/lcm/"+facet.getVariables().get(0).getName()+".obif", lcmResult.getBayesianNetwork());
        }

        /** Matrix with NID between facets */
        double[][] nidMatrix = new double[nonRepeatedDataFacets.size()][nonRepeatedDataFacets.size()];
        for(int i=0; i < nonRepeatedDataFacets.size(); i++)
            for(int j=0; j < nonRepeatedDataFacets.size(); j++)
                nidMatrix[i][j] = 0;

        for(int i=0; i < nonRepeatedDataFacets.size(); i++)
            for(int j=0; j < nonRepeatedDataFacets.size(); j++)
                if(!nonRepeatedDataFacets.get(i).equals(nonRepeatedDataFacets.get(j))){
                    // Each pair of facets creates a combined BN
                    Set<DiscreteVariable> nonRepeatedVariables = new LinkedHashSet<>();
                    nonRepeatedVariables.addAll(nonRepeatedDataFacets.get(i).getVariables());
                    nonRepeatedVariables.addAll(nonRepeatedDataFacets.get(j).getVariables());
                    List<DiscreteVariable> nonRepeatedVariablesList = new ArrayList<>(nonRepeatedVariables);
                    DiscreteBayesNet combinedBn = loadOrLearnCombinedFacetBn(nonRepeatedVariablesList, dataWeka);

                    // The distance between each pair is calculated
                    double nid = NID.calculate(nonRepeatedDataFacets.get(i).getBayesNet(), nonRepeatedDataFacets.get(j).getBayesNet(), combinedBn);
                    nidMatrix[i][j] = nid;
                }

        // Export generated matrix in CSV format
        ExportMatrixCSV.export("estudios/"+dataName+"/facets/lcm/nidMatrix.csv", nidMatrix);

        double nbTimeEnd = System.currentTimeMillis();
        double time = nbTimeEnd - nbTimeStart;

        System.out.println("Attribute clustering execution time: " + time);
    }

    /** Filtramos aquellas que estan compuestas por una única variable o estan repetidas */
    private static List<DataFacet> filterDataFacets(List<DataFacet> facets) {

        Set<DataFacet> nonRepeatedDataFacets = new LinkedHashSet<>();

        for(DataFacet facet: facets){
            if(facet.getVariables().size() > 1)
                nonRepeatedDataFacets.add(facet);
        }

        return new ArrayList<>(nonRepeatedDataFacets);

    }

    private static DiscreteBayesNet loadOrLearnCombinedFacetBn(List<DiscreteVariable> nonRepeatedVariables, Instances dataWeka) throws Exception {

        long hash = SimpleHashCreator.createHash(nonRepeatedVariables);
        if(loadedBns.containsKey(hash)){
           return loadedBns.get(hash);
        }

        DiscreteBayesNet learnedBn = codigo.learnBnUsingWeka(nonRepeatedVariables, dataWeka);
        loadedBns.put(SimpleHashCreator.createHash(nonRepeatedVariables), learnedBn);

        return learnedBn;
    }
}
