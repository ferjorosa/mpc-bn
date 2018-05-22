package experiment;

import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;
import weka.bif.XmlBifReader;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 16/05/2018.
 */
public class SyntheticExperiment {

    public static void main(String[] args) throws Exception {
        //run1Time();
        run5Times();
    }

    private static void run1Time() throws Exception {
        // Cargamos los datos de test
        DiscreteData syntheticCond10 = DataFileLoader.loadDiscreteData("synthetic/data/olcm10MVs_test_5000.arff");
        DiscreteData syntheticParkinson10 = DataFileLoader.loadDiscreteData("synthetic/data/olcm10MVs_parkinson_test_5000.arff");
        DiscreteData syntheticEgypt15 = DataFileLoader.loadDiscreteData("synthetic/data/olcm15MVs_egypt_test_5000.arff");

        /************************************** Cargamos los OLCMs originales */
        List<String> originalCondvida10LVs = new ArrayList<>();
        originalCondvida10LVs.add("variable69");
        originalCondvida10LVs.add("variable145");
        originalCondvida10LVs.add("variable11");
        DiscreteBayesNet originalCondvida10 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/olcm_condvida_10.xml"), originalCondvida10LVs);

        List<String> originalParkinson10LVs = new ArrayList<>();
        originalParkinson10LVs.add("variable211");
        originalParkinson10LVs.add("variable11");
        originalParkinson10LVs.add("variable66");
        originalParkinson10LVs.add("variable278");
        DiscreteBayesNet originalParkinson10 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/olcm_parkinson_10.xml"), originalParkinson10LVs);

        List<String> originalEgypt15LVs = new ArrayList<>();
        originalEgypt15LVs.add("variable16");
        originalEgypt15LVs.add("variable98");
        originalEgypt15LVs.add("variable243");
        DiscreteBayesNet originalEgypt15 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/olcm_egypt_15.xml"), originalEgypt15LVs);

        /************************************** Cargamos los OLCMs aprendidos */
        List<String> learnedCondvida10LVs = new ArrayList<>();
        learnedCondvida10LVs.add("variable81");
        learnedCondvida10LVs.add("variable260");
        learnedCondvida10LVs.add("variable304");
        DiscreteBayesNet learnedCondvida10 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/synthetic_condvida_10_olcm.xml"), learnedCondvida10LVs);

        List<String> learnedParkinson10LVs = new ArrayList<>();
        learnedParkinson10LVs.add("variable126");
        learnedParkinson10LVs.add("variable205");
        learnedParkinson10LVs.add("variable71");
        learnedParkinson10LVs.add("variable284");
        DiscreteBayesNet learnedParkinson10 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/synthetic_parkinson_10_olcm.xml"), learnedParkinson10LVs);

        List<String> learnedEgypt15LVs = new ArrayList<>();
        learnedEgypt15LVs.add("variable1797");
        learnedEgypt15LVs.add("variable686");
        learnedEgypt15LVs.add("variable859");
        DiscreteBayesNet learnedEgypt15 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/synthetic_egypt_15_olcm.xml"), learnedEgypt15LVs);

        /************************************** Comparamos sus scores de LL sobre los datos de test */
        // CondVida
        System.out.println("\n\n CondVida data");
        double condvidaOriginalLL = LearningScore.calculateLogLikelihood(syntheticCond10, originalCondvida10);
        double condvidaLearnedLL = LearningScore.calculateLogLikelihood(syntheticCond10, learnedCondvida10);
        System.out.println("Original LL: "+ condvidaOriginalLL);
        System.out.println("Learned LL: " + condvidaLearnedLL);
        System.out.println("KL divergencia empirica: "+ (condvidaOriginalLL - condvidaLearnedLL)/5000);

        // Parkinson
        System.out.println("\n\n Parkinson data");
        double parkinsonOriginalLL = LearningScore.calculateLogLikelihood(syntheticParkinson10, originalParkinson10);
        double parkinsonLearnedLL = LearningScore.calculateLogLikelihood(syntheticParkinson10, learnedParkinson10);
        System.out.println("Original LL: "+ parkinsonOriginalLL);
        System.out.println("Learned LL: " + parkinsonLearnedLL);
        System.out.println("KL divergencia empirica: "+ (parkinsonOriginalLL - parkinsonLearnedLL)/5000);

        // Egypt
        System.out.println("\n\n Egypt data");
        double egyptOriginalLL = LearningScore.calculateLogLikelihood(syntheticEgypt15, originalEgypt15);
        double egyptLearnedLL = LearningScore.calculateLogLikelihood(syntheticEgypt15, learnedEgypt15);
        System.out.println("Original LL: " + egyptOriginalLL);
        System.out.println("Learned LL: " + egyptLearnedLL);
        System.out.println("KL divergencia empirica: "+ (egyptOriginalLL - egyptLearnedLL)/5000);
    }

    private static void run5Times() throws Exception {

        // Cargamos los datos de learn
        DiscreteData syntheticCond10_train = DataFileLoader.loadDiscreteData("synthetic/data/olcm10MVs_test_5000.arff");
        DiscreteData syntheticParkinson10_train = DataFileLoader.loadDiscreteData("synthetic/data/olcm10MVs_parkinson_test_5000.arff");
        DiscreteData syntheticEgypt15_train = DataFileLoader.loadDiscreteData("synthetic/data/olcm15MVs_egypt_test_5000.arff");

        // Cargamos los datos de test
        DiscreteData syntheticCond10_test = DataFileLoader.loadDiscreteData("synthetic/data/olcm10MVs_test_5000.arff");
        DiscreteData syntheticParkinson10_test = DataFileLoader.loadDiscreteData("synthetic/data/olcm10MVs_parkinson_test_5000.arff");
        DiscreteData syntheticEgypt15_test = DataFileLoader.loadDiscreteData("synthetic/data/olcm15MVs_egypt_test_5000.arff");

        /************************************** Cargamos los OLCMs originales */
        List<String> originalCondvida10LVs = new ArrayList<>();
        originalCondvida10LVs.add("variable69");
        originalCondvida10LVs.add("variable145");
        originalCondvida10LVs.add("variable11");
        DiscreteBayesNet originalCondvida10 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/olcm_condvida_10.xml"), originalCondvida10LVs);

        List<String> originalParkinson10LVs = new ArrayList<>();
        originalParkinson10LVs.add("variable211");
        originalParkinson10LVs.add("variable11");
        originalParkinson10LVs.add("variable66");
        originalParkinson10LVs.add("variable278");
        DiscreteBayesNet originalParkinson10 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/olcm_parkinson_10.xml"), originalParkinson10LVs);

        List<String> originalEgypt15LVs = new ArrayList<>();
        originalEgypt15LVs.add("variable16");
        originalEgypt15LVs.add("variable98");
        originalEgypt15LVs.add("variable243");
        DiscreteBayesNet originalEgypt15 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/olcm_egypt_15.xml"), originalEgypt15LVs);

        /************************************** Cargamos los OLCMs aprendidos */
        List<String> learnedCondvida10LVs = new ArrayList<>();
        learnedCondvida10LVs.add("variable81");
        learnedCondvida10LVs.add("variable260");
        learnedCondvida10LVs.add("variable304");
        DiscreteBayesNet learnedCondvida10 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/synthetic_condvida_10_olcm.xml"), learnedCondvida10LVs);

        List<String> learnedParkinson10LVs = new ArrayList<>();
        learnedParkinson10LVs.add("variable126");
        learnedParkinson10LVs.add("variable205");
        learnedParkinson10LVs.add("variable71");
        learnedParkinson10LVs.add("variable284");
        DiscreteBayesNet learnedParkinson10 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/synthetic_parkinson_10_olcm.xml"), learnedParkinson10LVs);

        List<String> learnedEgypt15LVs = new ArrayList<>();
        learnedEgypt15LVs.add("variable1797");
        learnedEgypt15LVs.add("variable686");
        learnedEgypt15LVs.add("variable859");
        DiscreteBayesNet learnedEgypt15 = voltric.io.model.xmlbif.XmlBifReader.processFile(new File("synthetic/synthetic_egypt_15_olcm.xml"), learnedEgypt15LVs);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        int nRuns = 4;
        int currentRun = 1;
        // CondVida
        System.out.println("\n\n CondVida data");
        while(currentRun <= nRuns) {
            currentRun++;
            System.out.println("Run " + currentRun + "\n");
            LearningResult<DiscreteBayesNet> learnedCondvida10result = em.learnModel(learnedCondvida10, syntheticCond10_train);
            double condvidaOriginalLL = LearningScore.calculateLogLikelihood(syntheticCond10_test, originalCondvida10);
            double condvidaLearnedLL = LearningScore.calculateLogLikelihood(syntheticCond10_test, learnedCondvida10result.getBayesianNetwork());
            System.out.println("Original LL: "+ condvidaOriginalLL);
            System.out.println("Learned LL: " + condvidaLearnedLL);
            System.out.println("KL divergencia empirica: "+ (condvidaOriginalLL - condvidaLearnedLL)/5000);
        }

        int nRunsB = 4;
        int currentRunB = 1;
        // Parkinson
        System.out.println("\n\n Parkinson data");
        while(currentRunB <= nRunsB) {
            currentRunB++;
            System.out.println("Run " + currentRunB + "\n");
            LearningResult<DiscreteBayesNet> learnedParkinsonResult = em.learnModel(learnedParkinson10, syntheticParkinson10_train);
            double parkinsonOriginalLL = LearningScore.calculateLogLikelihood(syntheticParkinson10_test, originalParkinson10);
            double parkinsonLearnedLL = LearningScore.calculateLogLikelihood(syntheticParkinson10_test, learnedParkinsonResult.getBayesianNetwork());
            System.out.println("Original LL: "+ parkinsonOriginalLL);
            System.out.println("Learned LL: " + parkinsonLearnedLL);
            System.out.println("KL divergencia empirica: "+ (parkinsonOriginalLL - parkinsonLearnedLL)/5000);
        }

        int nRunsC = 4;
        int currentRunC = 1;
        // CondVida
        System.out.println("\n\n Egypt data");
        while(currentRunC <= nRunsC) {
            currentRunC++;
            System.out.println("Run " + currentRunC + "\n");
            LearningResult<DiscreteBayesNet> learnedEgyptResult = em.learnModel(learnedEgypt15, syntheticEgypt15_train);
            double egyptOriginalLL = LearningScore.calculateLogLikelihood(syntheticEgypt15_test, originalEgypt15);
            double egyptLearnedLL = LearningScore.calculateLogLikelihood(syntheticEgypt15_test, learnedEgyptResult.getBayesianNetwork());
            System.out.println("Original LL: "+ egyptOriginalLL);
            System.out.println("Learned LL: " + egyptLearnedLL);
            System.out.println("KL divergencia empirica: "+ (egyptOriginalLL - egyptLearnedLL)/5000);
        }
    }
}
