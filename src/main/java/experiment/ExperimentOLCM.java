package experiment;

import articulo.learning.HellingerFinder;
import articulo.learning.OlcmHillClimbing;
import articulo.learning.olcm.operator.*;
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
import java.util.HashSet;
import java.util.Set;

/**
 * Created by equipo on 27/04/2018.
 */
public class ExperimentOLCM {

    public static void main(String[] args) throws Exception {

        /** 10MVs synthetic data */
        String olcm10MVs_train_string = "experiments/synthetic/10MVs/data/olcm10MVs_train_5000.arff";
        String olcm10MVs_test_string = "experiments/synthetic/10MVs/data/olcm10MVs_test_5000.arff";
        DiscreteData olcm10MVs_train = DataFileLoader.loadDiscreteData(olcm10MVs_train_string);
        DiscreteData olcm10MVs_test = DataFileLoader.loadDiscreteData(olcm10MVs_test_string);

        /** 15 MVs synthetic parkinson data */
        /*
        String olcm15MVs_parkinson_train_string = "experiments/synthetic/15MVs/data/olcm15MVs_parkinson_2_train_5000.arff";
        String olcm15MVs_parkinson_test_string = "experiments/synthetic/15MVs/data/olcm15MVs_parkinson_2_test_5000.arff";
        DiscreteData olcm15MVs_parkinson_train = DataFileLoader.loadDiscreteData(olcm15MVs_parkinson_train_string);
        DiscreteData olcm15MVs_parkinson_test = DataFileLoader.loadDiscreteData(olcm15MVs_parkinson_test_string);
        */

        /** 15 MVs synthetic parkinson data */
        String olcm10MVs_parkinson_train_string = "experiments/synthetic/10MVs/data/olcm10MVs_parkinson_train_5000.arff";
        String olcm10MVs_parkinson_test_string = "experiments/synthetic/10MVs/data/olcm10MVs_parkinson_test_5000.arff";
        DiscreteData olcm10MVs_parkinson_train = DataFileLoader.loadDiscreteData(olcm10MVs_parkinson_train_string);
        DiscreteData olcm10MVs_parkinson_test = DataFileLoader.loadDiscreteData(olcm10MVs_parkinson_test_string);

        /** 15 MVs synthetic egypt data */
        String olcm15MVs_egypt_train_string = "experiments/synthetic/15MVs/data/olcm15MVs_egypt_train_5000.arff";
        String olcm15MVs_egypt_test_string = "experiments/synthetic/15MVs/data/olcm15MVs_egypt_test_5000.arff";
        DiscreteData olcm15MVs_egypt_train = DataFileLoader.loadDiscreteData(olcm15MVs_egypt_train_string);
        DiscreteData olcm15MVs_egypt_test = DataFileLoader.loadDiscreteData(olcm15MVs_egypt_test_string);

        /** 12 MVs real egypt data */
        String olcm_egypt_12_train_string = "experiments/real/data/egypt_12.arff";
        DiscreteData olcm_egypt_12_train = DataFileLoader.loadDiscreteData(olcm_egypt_12_train_string);

        /** 12 MVs real egypt data V2*/
        String olcm_egypt_12_V2_train_string = "experiments/real/data/egypt_12_v2.arff";
        DiscreteData olcm_egypt_12_v2_train = DataFileLoader.loadDiscreteData(olcm_egypt_12_train_string);

        /** 12 MVs real egypt data V3*/
        String olcm_egypt_12_v3_train_string = "experiments/real/data/egypt_12_v3.arff";
        DiscreteData olcm_egypt_12_v3_train = DataFileLoader.loadDiscreteData(olcm_egypt_12_train_string);

        /** 12 MVs real cond_vida data */
        String olcm_condVida_12_train_string = "experiments/real/data/cond_12.arff";
        DiscreteData olcm_condVida_12_train = DataFileLoader.loadDiscreteData(olcm_egypt_12_train_string);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        int nRuns = 1;
        int currentRun = 1;

        while(currentRun <= nRuns) {
            System.out.println("Run " + currentRun + "\n");

            //learnCondVida(olcm10MVs_train, olcm10MVs_test, em, currentRun);
            //learnEgypt(olcm15MVs_egypt_train, olcm15MVs_egypt_test, em, currentRun);
            //learnParkinson(olcm10MVs_parkinson_train, olcm10MVs_parkinson_test, em, currentRun);
            //learnRealEgypt(olcm_egypt_12_train, em, currentRun);
            learnRealCondVida(olcm_condVida_12_train, em, currentRun);
            learnRealEgypt_v2(olcm_egypt_12_v2_train, em, currentRun);
            learnRealEgypt_v3(olcm_egypt_12_v3_train, em, currentRun);
            currentRun++;
        }
    }

    private static void learnEgypt(DiscreteData egypt_learn_data, DiscreteData egypt_test_data, AbstractEM em, int currentRun) throws Exception {

        System.out.println("\n 15 MVs Egypt data \n");

        double initTime = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM = HellingerFinder.find(egypt_learn_data, em);
        LearningResult<DiscreteBayesNet> olcm = apply6operators(initialOLCM, egypt_learn_data, "experiments/synthetic/15MVs/OLCM/egypt", em);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(egypt_learn_data, olcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ olcm.getScoreValue());

        double llTestScore = LearningScore.calculateLogLikelihood(egypt_test_data, olcm.getBayesianNetwork());

        System.out.println("Learning time: " + (endTime-initTime)+ " ms");
        System.out.println("LL test score: "+ llTestScore);
        System.out.print("BIC test score: "+ LearningScore.calculateBIC(egypt_test_data, olcm.getBayesianNetwork(), llTestScore));
        System.out.println("\n\n");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/synthetic/15MVs/OLCM/egypt/olcm_egypt_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm.getBayesianNetwork());
    }

    private static void learnParkinson(DiscreteData parkinson_learn_data, DiscreteData parkinson_test_data, AbstractEM em, int currentRun) throws Exception {

        System.out.println("\n 10 MVs Parkinson data \n");

        double initTime = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM = HellingerFinder.find(parkinson_learn_data, em);
        LearningResult<DiscreteBayesNet> olcm = apply6operators(initialOLCM, parkinson_learn_data, "experiments/synthetic/10MVs/OLCM/parkinson", em);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(parkinson_learn_data, olcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ olcm.getScoreValue());

        double llTestScore = LearningScore.calculateLogLikelihood(parkinson_test_data, olcm.getBayesianNetwork());

        System.out.println("Learning time: " + (endTime-initTime)+ " ms");
        System.out.println("LL test score: "+ llTestScore);
        System.out.print("BIC test score: "+ LearningScore.calculateBIC(parkinson_test_data, olcm.getBayesianNetwork(), llTestScore));
        System.out.println("\n\n");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/synthetic/10MVs/OLCM/parkinson/olcm_parkinson_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm.getBayesianNetwork());
    }

    private static void learnCondVida(DiscreteData condvida_learn_data, DiscreteData condvida_test_data, AbstractEM em, int currentRun) throws Exception {

        System.out.println("\n 10 MVs Condiciones_vida data \n");

        double initTime = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM = HellingerFinder.find(condvida_learn_data, em);
        LearningResult<DiscreteBayesNet> olcm = apply6operators(initialOLCM, condvida_learn_data, "experiments/synthetic/10MVs/OLCM/cond_vida", em);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(condvida_learn_data, olcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ olcm.getScoreValue());

        double llTestScore = LearningScore.calculateLogLikelihood(condvida_test_data, olcm.getBayesianNetwork());

        System.out.println("Learning time: " + (endTime-initTime)+ " ms");
        System.out.println("LL test score: "+ llTestScore);
        System.out.print("BIC test score: "+ LearningScore.calculateBIC(condvida_test_data, olcm.getBayesianNetwork(), llTestScore));
        System.out.println("\n\n");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/synthetic/10MVs/OLCM/cond_vida/olcm_cond_vida_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm.getBayesianNetwork());
    }

    private static void learnRealEgypt(DiscreteData egypt_real_learn_data, AbstractEM em, int currentRun) throws Exception{
        System.out.println("\n 12 MVs Real Egypt data \n");

        double initTime = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM = HellingerFinder.find(egypt_real_learn_data, em);
        LearningResult<DiscreteBayesNet> olcm = apply6operators(initialOLCM, egypt_real_learn_data, "experiments/real/OLCM/egypt", em);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(egypt_real_learn_data, olcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ olcm.getScoreValue());
        System.out.println("Learning time: "+ (endTime - initTime) + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/real/OLCM/egypt/olcm_egypt_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm.getBayesianNetwork());
    }

    private static void learnRealEgypt_v2(DiscreteData egypt_real_learn_data, AbstractEM em, int currentRun) throws Exception{
        System.out.println("\n 12 MVs Real Egypt data V2 \n");

        double initTime = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM = HellingerFinder.find(egypt_real_learn_data, em);
        LearningResult<DiscreteBayesNet> olcm = apply6operators(initialOLCM, egypt_real_learn_data, "experiments/real/OLCM/egypt_v2", em);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(egypt_real_learn_data, olcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ olcm.getScoreValue());
        System.out.println("Learning time: "+ (endTime - initTime) + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/real/OLCM/egypt_v2/olcm_egypt_v2_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm.getBayesianNetwork());
    }

    private static void learnRealEgypt_v3(DiscreteData egypt_real_learn_data, AbstractEM em, int currentRun) throws Exception{
        System.out.println("\n 12 MVs Real Egypt data V3 \n");

        double initTime = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM = HellingerFinder.find(egypt_real_learn_data, em);
        LearningResult<DiscreteBayesNet> olcm = apply6operators(initialOLCM, egypt_real_learn_data, "experiments/real/OLCM/egypt_v3", em);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(egypt_real_learn_data, olcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ olcm.getScoreValue());
        System.out.println("Learning time: "+ (endTime - initTime) + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/real/OLCM/egypt_v3/olcm_egypt_v3_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm.getBayesianNetwork());
    }

    private static void learnRealCondVida(DiscreteData condVida_real_learn_data, AbstractEM em, int currentRun) throws Exception{
        System.out.println("\n 12 MVs Real CondVida data \n");

        double initTime = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM = HellingerFinder.find(condVida_real_learn_data, em);
        LearningResult<DiscreteBayesNet> olcm = apply6operators(initialOLCM, condVida_real_learn_data, "experiments/real/OLCM/cond", em);
        double endTime = System.currentTimeMillis();

        System.out.println("Learning LL: " + LearningScore.calculateLogLikelihood(condVida_real_learn_data, olcm.getBayesianNetwork()));
        System.out.println("Learning BIC: "+ olcm.getScoreValue());
        System.out.println("Learning time: "+ (endTime - initTime) + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("experiments/real/OLCM/cond/olcm_condVida_run"+currentRun+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm.getBayesianNetwork());
    }

    private static LearningResult<DiscreteBayesNet> apply6operators(DiscreteBayesNet bn, DiscreteData data, String dataStore, AbstractEM em) throws Exception {


        /** Expansion Operators */
        Set<OlcmHcOperator> expansionOperators = new HashSet<>();
        expansionOperators.add(new AddOlcmArc());
        expansionOperators.add(new NewAddOlcmNode());
        expansionOperators.add(new IncreaseOlcmCard(10));

        /** Simplification Operators */
        Set<OlcmHcOperator> simplificationOperators = new HashSet<>();
        simplificationOperators.add(new RemoveOlcmArc());
        simplificationOperators.add(new NewRemoveOlcmNode());
        simplificationOperators.add(new DecreaseOlcmCard(2));

        OlcmHillClimbing hillClimbing = new OlcmHillClimbing(400, 0.5, expansionOperators, simplificationOperators);

        return hillClimbing.learnModel(bn, data, dataStore, em);
    }
}

