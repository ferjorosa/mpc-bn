package experiment;

import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 09/05/2018.
 *
 * Cargamos el modelo OLCM y utilizamos su estructura para aprender con el EM uno nuevo  el cuall exportamos en formatos
 * BIF y XML para su posterior analisis.
 */
public class LearnLoadedModel {

    public static void main(String[] args) throws Exception {

        //transformData();
        learnModel();
    }

    private static void learnModel() throws Exception{

        List<String> latentVars = new ArrayList<>();
        latentVars.add("variable1963");
        latentVars.add("variable2075");
        latentVars.add("variable2377");
        latentVars.add("variable681");
        //latentVars.add("variable50");

        // Load model
        DiscreteBayesNet bn = XmlBifReader.processFile(new File("OLHC_real_egypt_v3_base.xml"), latentVars);
        // Load data
        /** 12 MVs real cond_vida data */
        String data_string = "experiments/real/data/egypt_12_v3.arff";
        DiscreteData data = DataFileLoader.loadDiscreteData(data_string);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);
        LearningResult<DiscreteBayesNet> olcm =  em.learnModel(bn, data);

        // Show score
        System.out.println("BIC: " + olcm.getScoreValue());
        System.out.println("LL: " + LearningScore.calculateLogLikelihood(data, bn));

        // Calculate BIC score old model
        //System.out.println("BIC: " + LearningScore.calculateBIC(data, bn));

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("OLHC_real_egypt_v3_run"+5+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm.getBayesianNetwork());
    }
}
