package example;

import voltric.clustering.singleview.HiddenNaiveBayes;
import voltric.clustering.singleview.LatentTAN;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.util.stattest.discrete.MutualInformation;

import java.io.FileOutputStream;
import java.io.OutputStream;

public class LearnTANTest {

    public static void main(String[] args) throws Exception {
        String asia_train_string = "estudios/Asia_train.arff";
        String parkinson_string = "estudios/parkinson/data/d897_motor25.arff";
        DiscreteData data = DataFileLoader.loadDiscreteData(parkinson_string);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        int nRuns = 5;
        int currentRun = 1;

        while(currentRun <= nRuns) {
            System.out.println("Run " + currentRun + "\n");

            System.out.println("\n\nLCM:\n");

            /** Learn LCM */
            LearningResult<DiscreteBayesNet> lcm = learnLCM(data, em);
            // Export model in BIF 0.15 format
            OutputStream nbOutput1 = new FileOutputStream("pruebasTAN/lcm_parkinson_motor_run"+currentRun+".bif");
            BnLearnBifFileWriter writerLCM = new BnLearnBifFileWriter(nbOutput1);
            writerLCM.write(lcm.getBayesianNetwork());

            System.out.println("\n\nTAN:\n");

            /** Learn TAN */
            LearningResult<DiscreteBayesNet> tan = learnTAN(data, em);
            // Export model in BIF 0.15 format
            OutputStream nbOutput2 = new FileOutputStream("pruebasTAN/tan_parkinson_motor_run"+currentRun+".bif");
            BnLearnBifFileWriter writerTAN = new BnLearnBifFileWriter(nbOutput2);
            writerTAN.write(tan.getBayesianNetwork());

            currentRun++;
        }
    }

    private static LearningResult<DiscreteBayesNet> learnLCM(DiscreteData data, AbstractEM em) {
        double initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> lcm = HiddenNaiveBayes.learnModel(25, data, em, 0.5);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(data, lcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ lcm.getScoreValue());
        System.out.println("Learning time: " + (endTime-initTime)+ " ms");

        return lcm;
    }

    private static LearningResult<DiscreteBayesNet> learnTAN(DiscreteData data, AbstractEM em) {
        double initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> tan = LatentTAN.learnModel(25, data, em, 0.5, new MutualInformation());
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(data, tan.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ tan.getScoreValue());
        System.out.println("Learning time: " + (endTime-initTime)+ " ms");

        return tan;
    }
}
