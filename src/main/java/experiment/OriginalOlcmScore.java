package experiment;

import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.bif.OldBifFileReader;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;
import weka.bif.XmlBifReader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 25/04/2018.
 *
 * Cargamos la red que origino los datos en formato OBIf y calculamos sus scores LL y BIC para los datos de test,
 * con objetivo de que nos sirva en las comparativas
 */
public class OriginalOlcmScore {

    public static void main(String[] args) throws Exception {

        /** OLCM 10 MVs data */
        String olcm10MVs_test_string = "experiments/synthetic/10MVs/data/olcm10MVs_test_20000.arff";
        String olcm10MVs_train_string = "experiments/synthetic/10MVs/data/olcm10MVs_train_20000.arff";
        DiscreteData olcm10MVs_test = DataFileLoader.loadDiscreteData(olcm10MVs_test_string);
        DiscreteData olcm10MVs_train = DataFileLoader.loadDiscreteData(olcm10MVs_train_string);

        /** OLCM 15 MVs parkinson data */
        String olcm15MVs_train_string = "experiments/synthetic/15MVs/data/olcm15MVs_parkinson_2_train_5000.arff";
        String olcm15MVs_test_string = "experiments/synthetic/15MVs/data/olcm15MVs_parkinson_2_test_5000.arff";
        DiscreteData olcm15MVs_train = DataFileLoader.loadDiscreteData(olcm15MVs_train_string);
        DiscreteData olcm15MVs_test = DataFileLoader.loadDiscreteData(olcm15MVs_test_string);
/*
        DiscreteBayesNet originalOLCM_10MVs = OldBifFileReader.readOBif("experiments/synthetic/10MVs/olcm_10MVs.obif");
        DiscreteBayesNet originalOLCM_15MVs = OldBifFileReader.readOBif("experiments/synthetic/15MVs/olcm_15MVs.obif");

        double llScore_10MVs = LearningScore.calculateLogLikelihood(olcm10MVs_test, originalOLCM_10MVs);
        double bicScore_10MVs = LearningScore.calculateBIC(olcm10MVs_test, originalOLCM_10MVs, llScore_10MVs);

        double llScore_15MVs = LearningScore.calculateLogLikelihood(olcm15MVs_test, originalOLCM_15MVs);
        double bicScore_15MVs = LearningScore.calculateBIC(olcm15MVs_test, originalOLCM_15MVs, llScore_15MVs);

        System.out.println("10 MVS: ");
        System.out.println("LL: " + llScore_10MVs);
        System.out.println("BIC: " + bicScore_10MVs);

        System.out.println("15 MVS: ");
        System.out.println("LL: " + llScore_15MVs);
        System.out.println("BIC: " + bicScore_15MVs);
*/
        //testOriginalModel(olcm10MVs_train, olcm10MVs_test);
        //learnOriginalModel(olcm10MVs_train, olcm10MVs_test);
        //learnBI(olcm10MVs_train, olcm10MVs_test);
        //testOriginalModelParkinson15MVs(olcm15MVs_train, olcm15MVs_test);
    }

    private static void testOriginalModel10MVs(DiscreteData train_data, DiscreteData test_data) throws Exception{
        List<String> latentVarNames = new ArrayList<>();
        latentVarNames.add("variable69"); latentVarNames.add("variable145"); latentVarNames.add("variable11");
        DiscreteBayesNet loadedBn = XmlBifReader.processFile(new File("experiments/synthetic/10MVs/olcm_10MVs.xml"), latentVarNames);

        double testLL = LearningScore.calculateLogLikelihood(test_data, loadedBn);
        double trainLL = LearningScore.calculateLogLikelihood(train_data, loadedBn);

        System.out.println("Train LL: "+ trainLL);
        System.out.println("Test LL: "+testLL);
    }

    private static void testOriginalModelParkinson15MVs(DiscreteData train_data, DiscreteData test_data) throws Exception {
        List<String> latentVarNames = new ArrayList<>();
        latentVarNames.add("variable673"); latentVarNames.add("variable16"); latentVarNames.add("variable101"); latentVarNames.add("variable369");
        DiscreteBayesNet loadedBn = XmlBifReader.processFile(new File("experiments/synthetic/15MVs/olcm_15MVs_parkinson_2.xml"), latentVarNames);

        double trainLL = LearningScore.calculateLogLikelihood(train_data, loadedBn);
        double trainBIC = LearningScore.calculateBIC(train_data, loadedBn, trainLL);

        double testLL = LearningScore.calculateLogLikelihood(test_data, loadedBn);
        double testBIC = LearningScore.calculateBIC(test_data, loadedBn, testLL);

        System.out.println("Train LL: "+ trainLL);
        System.out.println("Train BIC: "+ trainBIC);
        System.out.println("Test LL: "+testLL);
        System.out.println("Test BIC: "+testBIC);
    }

    private static void learnOriginalModel(DiscreteData train_data, DiscreteData test_data) throws Exception{

        // Creamos 3 LVs
        DiscreteVariable variable69 = new DiscreteVariable(2, VariableType.LATENT_VARIABLE);
        DiscreteVariable variable145 = new DiscreteVariable(2, VariableType.LATENT_VARIABLE);
        DiscreteVariable variable11 = new DiscreteVariable(3, VariableType.LATENT_VARIABLE);

        // Creamos la BN y definimos su estructura

        DiscreteBayesNet olcm = new DiscreteBayesNet();
        olcm.addNode(variable69);
        olcm.addNode(variable11);
        olcm.addNode(variable145);

        for(DiscreteVariable mv: train_data.getVariables())
            olcm.addNode(mv);

        olcm.addEdge(olcm.getNode("hs160"), olcm.getNode(variable11));
        olcm.addEdge(olcm.getNode("hs160"), olcm.getNode(variable145));

        olcm.addEdge(olcm.getNode("hs170"), olcm.getNode(variable11));
        olcm.addEdge(olcm.getNode("hs170"), olcm.getNode(variable145));

        olcm.addEdge(olcm.getNode("hs180"), olcm.getNode(variable11));
        olcm.addEdge(olcm.getNode("hs180"), olcm.getNode(variable145));

        olcm.addEdge(olcm.getNode("hs190"), olcm.getNode(variable11));
        olcm.addEdge(olcm.getNode("hs190"), olcm.getNode(variable145));

        olcm.addEdge(olcm.getNode("hh040"), olcm.getNode(variable11));
        olcm.addEdge(olcm.getNode("hh040"), olcm.getNode(variable145));

        olcm.addEdge(olcm.getNode("hs050"), olcm.getNode(variable11));
        olcm.addEdge(olcm.getNode("hs050"), olcm.getNode(variable69));

        olcm.addEdge(olcm.getNode("hs060"), olcm.getNode(variable11));
        olcm.addEdge(olcm.getNode("hs060"), olcm.getNode(variable69));

        olcm.addEdge(olcm.getNode("hh050"), olcm.getNode(variable11));
        olcm.addEdge(olcm.getNode("hh050"), olcm.getNode(variable69));

        olcm.addEdge(olcm.getNode("h79_u"), olcm.getNode(variable11));

        olcm.addEdge(olcm.getNode("h80_u"), olcm.getNode(variable11));

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.LogLikelihood);
        for(int i = 0; i < 5; i++) {
            LearningResult<DiscreteBayesNet> originalOlcmResult = em.learnModel(olcm, train_data);

            double testLL = LearningScore.calculateLogLikelihood(test_data, originalOlcmResult.getBayesianNetwork());
            double trainLL = LearningScore.calculateLogLikelihood(train_data, originalOlcmResult.getBayesianNetwork());

            System.out.println("Train LL: "+ trainLL);
            System.out.println("Test LL: "+testLL);
        }
    }

    public static void learnBI(DiscreteData train_data, DiscreteData test_data) {
        DiscreteVariable variable66 = new DiscreteVariable(4, VariableType.LATENT_VARIABLE);
        DiscreteVariable variable54 = new DiscreteVariable(4, VariableType.LATENT_VARIABLE);

        // Creamos la BN y definimos su estructura

        DiscreteBayesNet olcm = new DiscreteBayesNet();
        olcm.addNode(variable66);
        olcm.addNode(variable54);

        for(DiscreteVariable mv: train_data.getVariables())
            olcm.addNode(mv);

        olcm.addEdge(olcm.getNode("hs050"), olcm.getNode(variable54));
        olcm.addEdge(olcm.getNode("hs060"), olcm.getNode(variable54));
        olcm.addEdge(olcm.getNode("hh050"), olcm.getNode(variable54));
        olcm.addEdge(olcm.getNode("h79_u"), olcm.getNode(variable54));
        olcm.addEdge(olcm.getNode("h80_u"), olcm.getNode(variable54));
        olcm.addEdge(olcm.getNode("hs160"), olcm.getNode(variable66));
        olcm.addEdge(olcm.getNode("hs170"), olcm.getNode(variable66));
        olcm.addEdge(olcm.getNode("hs180"), olcm.getNode(variable66));
        olcm.addEdge(olcm.getNode("hs190"), olcm.getNode(variable66));
        olcm.addEdge(olcm.getNode("hh040"), olcm.getNode(variable66));
        olcm.addEdge(olcm.getNode(variable66), olcm.getNode(variable54));

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.LogLikelihood);
        for(int i = 0; i < 5; i++) {
            LearningResult<DiscreteBayesNet> biHlcmResult = em.learnModel(olcm, train_data);

            double testLL = LearningScore.calculateLogLikelihood(test_data, biHlcmResult.getBayesianNetwork());
            double trainLL = LearningScore.calculateLogLikelihood(train_data, biHlcmResult.getBayesianNetwork());

            System.out.println("Train LL: "+ trainLL);
            System.out.println("Test LL: "+testLL);
        }
    }
}
