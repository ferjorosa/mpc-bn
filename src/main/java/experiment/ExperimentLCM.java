package experiment;

import articulo.learning.HellingerFinder;
import voltric.clustering.singleview.HiddenNaiveBayes;
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

import java.io.FileOutputStream;
import java.io.OutputStream;

/**
 * Aprende el LCM con los datos de training y evalúa su score con los de test
 */
public class ExperimentLCM {

    public static void main(String[] args) throws Exception {

        /** 10MVs synthetic data */
        String olcm10MVs_train_string = "experiments/synthetic/10MVs/data/olcm10MVs_train_5000.arff";
        String olcm10MVs_test_string = "experiments/synthetic/10MVs/data/olcm10MVs_test_5000.arff";
        DiscreteData olcm10MVs_train = DataFileLoader.loadDiscreteData(olcm10MVs_train_string);
        DiscreteData olcm10MVs_test = DataFileLoader.loadDiscreteData(olcm10MVs_test_string);

        /** 15 MVs synthetic parkinson data */
        String olcm15MVs_parkinson_train_string = "experiments/synthetic/15MVs/data/olcm15MVs_parkinson_2_train_5000.arff";
        String olcm15MVs_parkinson_test_string = "experiments/synthetic/15MVs/data/olcm15MVs_parkinson_2_test_5000.arff";
        DiscreteData olcm15MVs_parkinson_train = DataFileLoader.loadDiscreteData(olcm15MVs_parkinson_train_string);
        DiscreteData olcm15MVs_parkinson_test = DataFileLoader.loadDiscreteData(olcm15MVs_parkinson_test_string);

        /** 15 MVs synthetic egypt data */
        String olcm15MVs_egypt_train_string = "experiments/synthetic/15MVs/data/olcm15MVs_egypt_train_5000.arff";
        String olcm15MVs_egypt_test_string = "experiments/synthetic/15MVs/data/olcm15MVs_egypt_test_5000.arff";
        DiscreteData olcm15MVs_egypt_train = DataFileLoader.loadDiscreteData(olcm15MVs_egypt_train_string);
        DiscreteData olcm15MVs_egypt_test = DataFileLoader.loadDiscreteData(olcm15MVs_egypt_test_string);

        /** 12 MVs real egypt data V2*/
        String olcm_egypt_12_V2_train_string = "experiments/real/data/egypt_12_v2.arff";
        DiscreteData olcm_egypt_12_v2_train = DataFileLoader.loadDiscreteData(olcm_egypt_12_V2_train_string);

        /** 12 MVs real egypt data V3*/
        String olcm_egypt_12_v3_train_string = "experiments/real/data/egypt_12_v3.arff";
        DiscreteData olcm_egypt_12_v3_train = DataFileLoader.loadDiscreteData(olcm_egypt_12_v3_train_string);

        /** 12 MVs real cond_vida data */
        String olcm_condVida_12_train_string = "experiments/real/data/cond_12.arff";
        DiscreteData olcm_condVida_12_train = DataFileLoader.loadDiscreteData(olcm_condVida_12_train_string);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        int nRuns = 5;
        int currentRun = 1;

        while(currentRun <= nRuns) {
            System.out.println("Run " + currentRun + "\n");

            //Test data
            //learnCondVida(olcm10MVs_train, olcm10MVs_test, em, currentRun);
            //learnEgypt(olcm15MVs_egypt_train, olcm15MVs_egypt_test, em, currentRun);
            //learnParkinson(olcm15MVs_parkinson_train, olcm15MVs_parkinson_test, em, currentRun);

            //Real data
            learnRealEgypt_v2(olcm_egypt_12_v2_train, em, currentRun);
            learnRealEgypt_v3(olcm_egypt_12_v3_train, em, currentRun);
            learnRealCondVida(olcm_condVida_12_train, em, currentRun);

            currentRun++;

        }
    }

    private static void learnEgypt(DiscreteData egypt_learn_data, DiscreteData egypt_test_data, AbstractEM em, int currentRun) throws Exception {

        System.out.println("\n 15 MVs Egypt data \n");

        double initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> lcm = HiddenNaiveBayes.learnModel(25, egypt_learn_data, em, 0.5);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(egypt_learn_data, lcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ lcm.getScoreValue());

        double llTestScore = LearningScore.calculateLogLikelihood(egypt_test_data, lcm.getBayesianNetwork());

        System.out.println("Learning time: " + (endTime-initTime)+ " ms");
        System.out.println("LL test score: "+ llTestScore);
        System.out.print("BIC test score: "+ LearningScore.calculateBIC(egypt_test_data, lcm.getBayesianNetwork(), llTestScore));
        System.out.println("\n\n");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/synthetic/15MVs/LCM/egypt/lcm_egypt_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(lcm.getBayesianNetwork());
    }

    private static void learnParkinson(DiscreteData parkinson_learn_data, DiscreteData parkinson_test_data, AbstractEM em, int currentRun) throws Exception {

        System.out.println("\n 15 MVs Parkinson data \n");

        double initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> lcm = HiddenNaiveBayes.learnModel(25, parkinson_learn_data, em, 0.5);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(parkinson_learn_data, lcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ lcm.getScoreValue());

        double llTestScore = LearningScore.calculateLogLikelihood(parkinson_test_data, lcm.getBayesianNetwork());

        System.out.println("Learning time: " + (endTime-initTime)+ " ms");
        System.out.println("LL test score: "+ llTestScore);
        System.out.print("BIC test score: "+ LearningScore.calculateBIC(parkinson_test_data, lcm.getBayesianNetwork(), llTestScore));
        System.out.println("\n\n");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/synthetic/15MVs/LCM/parkinson/lcm_parkinson_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(lcm.getBayesianNetwork());
    }

    private static void learnCondVida(DiscreteData condvida_learn_data, DiscreteData condvida_test_data, AbstractEM em, int currentRun) throws Exception {

        System.out.println("\n 10 MVs Condiciones_vida data \n");

        double initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> lcm = HiddenNaiveBayes.learnModel(25, condvida_learn_data, em, 0.5);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(condvida_learn_data, lcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ lcm.getScoreValue());

        double llTestScore = LearningScore.calculateLogLikelihood(condvida_test_data, lcm.getBayesianNetwork());

        System.out.println("Learning time: " + (endTime-initTime)+ " ms");
        System.out.println("LL test score: "+ llTestScore);
        System.out.print("BIC test score: "+ LearningScore.calculateBIC(condvida_test_data, lcm.getBayesianNetwork(), llTestScore));
        System.out.println("\n\n");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/synthetic/10MVs/LCM/cond_vida/lcm_cond_vida_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(lcm.getBayesianNetwork());
    }

    private static void learnRealEgypt_v2(DiscreteData egypt_real_learn_data, AbstractEM em, int currentRun) throws Exception{
        System.out.println("\n 12 MVs Real Egypt data V2 \n");
        System.out.println("\nLearn LCM\n");

        double initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> lcm = HiddenNaiveBayes.learnModel(25, egypt_real_learn_data, em, 0.5);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(egypt_real_learn_data, lcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ lcm.getScoreValue());
        System.out.println("Learning time: "+ (endTime - initTime) + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/real/LCM/egypt_v2/lcm_egypt_v2_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(lcm.getBayesianNetwork());
    }

    private static void learnRealEgypt_v3(DiscreteData egypt_real_learn_data, AbstractEM em, int currentRun) throws Exception{
        System.out.println("\n 12 MVs Real Egypt data V3 \n");
        System.out.println("\nLearn LCM\n");

        double initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> lcm = HiddenNaiveBayes.learnModel(25, egypt_real_learn_data, em, 0.5);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(egypt_real_learn_data, lcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ lcm.getScoreValue());
        System.out.println("Learning time: "+ (endTime - initTime) + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/real/LCM/egypt_v3/lcm_egypt_v3_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(lcm.getBayesianNetwork());
    }

    private static void learnRealCondVida(DiscreteData condVida_real_learn_data, AbstractEM em, int currentRun) throws Exception{
        System.out.println("\n 12 MVs Real CondVida data \n");
        System.out.println("\nLearn LCM\n");

        double initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> lcm = HiddenNaiveBayes.learnModel(25, condVida_real_learn_data, em, 0.5);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(condVida_real_learn_data, lcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ lcm.getScoreValue());
        System.out.println("Learning time: "+ (endTime - initTime) + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/real/LCM/cond/lcm_condVida_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(lcm.getBayesianNetwork());
    }
}
