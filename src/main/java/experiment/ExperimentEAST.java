package experiment;

import articulo.scripts.util.GenerateDataEAST;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 26/04/2018.
 *
 * Dado que la ejecucion del código no e puede realizar aqui, por ahora simplemente vamos a implementar la transformacion
 * de dataset ARFF a EAST
 *
 * TODO: Probar a cargar un BIF 0.1 con JBayes para intentar que no tengamos que hacer el modelo a mano
 */
public class ExperimentEAST {

    public static void main(String[] args) throws Exception {

        //transformData();
        testModel();
    }

    private static void testModel() throws Exception{

        List<String> latentVars = new ArrayList<>();
        latentVars.add("variable12");
        latentVars.add("variable26");
        latentVars.add("variable36");
        latentVars.add("variable37");
        //latentVars.add("variable50");

        // Load model
        DiscreteBayesNet bn = XmlBifReader.processFile(new File("EAST_real_egypt_v3_hlcm_5.xml"), latentVars);
        // Load data
        /** 12 MVs real cond_vida data */
        String olcm_condVida_12_train_string = "experiments/real/data/egypt_12_v3.arff";
        DiscreteData olcm_condVida_12_train = DataFileLoader.loadDiscreteData(olcm_condVida_12_train_string);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        // Calculate BIC
        //System.out.println(em.learnModel(bn, olcm_egypt_12_v3_train).getScoreValue());
        System.out.println("LL: " + LearningScore.calculateLogLikelihood(olcm_condVida_12_train, bn));
        System.out.println("BIC: " + LearningScore.calculateBIC(olcm_condVida_12_train, bn));
    }

    private static void transformData() throws Exception{

        /** 12 MVs real egypt data V2*/
        String olcm_egypt_12_V2_train_string = "experiments/real/data/egypt_12_v2.arff";
        DiscreteData olcm_egypt_12_v2_train = DataFileLoader.loadDiscreteData(olcm_egypt_12_V2_train_string);

        /** 12 MVs real egypt data V3*/
        String olcm_egypt_12_v3_train_string = "experiments/real/data/egypt_12_v3.arff";
        DiscreteData olcm_egypt_12_v3_train = DataFileLoader.loadDiscreteData(olcm_egypt_12_v3_train_string);

        /** 12 MVs real cond_vida data */
        String olcm_condVida_12_train_string = "experiments/real/data/cond_12.arff";
        DiscreteData olcm_condVida_12_train = DataFileLoader.loadDiscreteData(olcm_condVida_12_train_string);

        // Export data in EAST format
        GenerateDataEAST.generate(olcm_condVida_12_train, "cond_12.data");
        GenerateDataEAST.generate(olcm_egypt_12_v2_train, "egypt_12_v2.data");
        GenerateDataEAST.generate(olcm_egypt_12_v3_train, "egypt_12_v3.data");
    }
}
