package example;

import articulo.learning.HellingerFinder;
import articulo.learning.OlcmHillClimbing;
import articulo.learning.olcm.operator.*;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.bif.OldBifFileWriter;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by equipo on 20/04/2018.
 *
 * TODO: Comparamos los modelos obtenidos por LCM, BI, EAST, 2HC con los OLCM
 *
 * BI: Numero de particiones: 4;  Cardinalidades: {2,2,3,3}; Score (BIC): -40136; Score (LL): -39949
 * LCM: Numero de particiones: 1; Cardinalidades: {5}; Score (BIC): -40261; Score(LL): -39930
 * InitOLCM: Numero de particiones: 3; Cardinalidades: {2,2,2}; Score (BIC): -42124
 */
public class OlcmModelCompare {

    public static void main(String[] args) throws Exception {
        // Cargamos los datos
        String condiciones_vida10vars = "estudios/condiciones_vida/data/condiciones_vida_10vars.arff";
        String parkison_15vars_v1 = "estudios/synthetic/15MVs_v1/data/egypt_15_v1.arff";
        String parkison_15vars_v2 = "estudios/synthetic/15MVs_v2/data/egypt_15_v2.arff";
        String parkison_15vars_v3 = "estudios/synthetic/15MVs_v3/data/egypt_15_v3.arff";

        /** Load Voltric data */

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);
        /*
        DiscreteData data = DataFileLoader.loadDiscreteData(condiciones_vida10vars);
        LearningResult<DiscreteBayesNet> lcm = HiddenNaiveBayes.learnModel(5, data, em, 0.5);
        System.out.println(lcm.getBayesianNetwork().toString());
        System.out.println("LCM: " + lcm.getScoreValue());
        */
/*
        String dataString_v1 = parkison_15vars_v1;
        DiscreteData dataVoltric_v1 = DataFileLoader.loadDiscreteData(dataString_v1);

        double initTime_v1 = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM_v1 = HellingerFinder.find(dataVoltric_v1, em);
        LearningResult<DiscreteBayesNet> olcmResult_v1 = apply4operators(initialOLCM_v1, dataVoltric_v1, em, "synthetic/15MVs_v1");
        double endTime_v1 = System.currentTimeMillis();
        System.out.println(olcmResult_v1.getBayesianNetwork().toString());
        System.out.println(olcmResult_v1.getScoreValue());
        System.out.println(endTime_v1 - initTime_v1 + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput_v1 = new FileOutputStream("olcmCompare_egypt15_v1.bif");
        BnLearnBifFileWriter writer_v1 = new BnLearnBifFileWriter(nbOutput_v1);
        writer_v1.write(olcmResult_v1.getBayesianNetwork());

        // Export model in OBIF format
        OldBifFileWriter.writeBif("olcmCompare_egypt15_v1.obif", olcmResult_v1.getBayesianNetwork());
*/
        /**************************************************************************************************************/
/*
        String dataString_v2 = parkison_15vars_v2;
        DiscreteData dataVoltric_v2 = DataFileLoader.loadDiscreteData(dataString_v2);

        double initTime_v2 = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM_v2 = HellingerFinder.find(dataVoltric_v2, em);
        LearningResult<DiscreteBayesNet> olcmResult_v2 = apply4operators(initialOLCM_v2, dataVoltric_v2, em, "synthetic/15MVs_v2");
        double endTime_v2 = System.currentTimeMillis();
        System.out.println(olcmResult_v2.getBayesianNetwork().toString());
        System.out.println(olcmResult_v2.getScoreValue());
        System.out.println(endTime_v2 - initTime_v2 + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput_v2 = new FileOutputStream("olcmCompare_egypt15_v2.bif");
        BnLearnBifFileWriter writer_v2 = new BnLearnBifFileWriter(nbOutput_v2);
        writer_v2.write(olcmResult_v2.getBayesianNetwork());

        // Export model in OBIF format
        OldBifFileWriter.writeBif("olcmCompare_egypt15_v2.obif", olcmResult_v2.getBayesianNetwork());
*/
        /**************************************************************************************************************/

        String dataString_v3 = parkison_15vars_v3;
        DiscreteData dataVoltric_v3 = DataFileLoader.loadDiscreteData(dataString_v3);

        double initTime_v3 = System.currentTimeMillis();
        DiscreteBayesNet initialOLCM_v3 = HellingerFinder.find(dataVoltric_v3, em);
        LearningResult<DiscreteBayesNet> olcmResult_v3 = apply4operators(initialOLCM_v3, dataVoltric_v3, em, "synthetic/15MVs_v3");
        double endTime_v3 = System.currentTimeMillis();
        System.out.println(olcmResult_v3.getBayesianNetwork().toString());
        System.out.println(olcmResult_v3.getScoreValue());
        System.out.println(endTime_v3 - initTime_v3 + " ms");

        // Export model in BIF 0.15 format
        OutputStream nbOutput_v3 = new FileOutputStream("olcmCompare_egypt15_v3.bif");
        BnLearnBifFileWriter writer_v3 = new BnLearnBifFileWriter(nbOutput_v3);
        writer_v3.write(olcmResult_v3.getBayesianNetwork());

        // Export model in OBIF format
        OldBifFileWriter.writeBif("olcmCompare_egypt15_v3.obif", olcmResult_v3.getBayesianNetwork());
    }

    private static LearningResult<DiscreteBayesNet> apply6operators(DiscreteBayesNet bn, DiscreteData data, AbstractEM em, String dataName) throws Exception {

        /** Expansion Operators */
        Set<OlcmHcOperator> expansionOperators = new HashSet<>();
        expansionOperators.add(new AddOlcmArc());
        expansionOperators.add(new NewAddOlcmNode());
        expansionOperators.add(new IncreaseOlcmCard(10));

        /** Simplification Operators */
        Set<OlcmHcOperator> simplificationOperators = new HashSet<>();
        simplificationOperators.add(new RemoveOlcmArc());
        simplificationOperators.add(new RemoveOlcmNode());
        simplificationOperators.add(new DecreaseOlcmCard(2));

        OlcmHillClimbing hillClimbing = new OlcmHillClimbing(400, 0.5, expansionOperators, simplificationOperators);

        return hillClimbing.learnModel(bn, data, dataName, em);
    }

    private static LearningResult<DiscreteBayesNet> apply4operators(DiscreteBayesNet bn, DiscreteData data, AbstractEM em, String dataName) throws Exception {

        /** Expansion Operators */
        Set<OlcmHcOperator> expansionOperators = new HashSet<>();
        expansionOperators.add(new AddOlcmArc());
        expansionOperators.add(new IncreaseOlcmCard(10));

        /** Simplification Operators */
        Set<OlcmHcOperator> simplificationOperators = new HashSet<>();
        simplificationOperators.add(new RemoveOlcmArc());
        simplificationOperators.add(new DecreaseOlcmCard(2));

        OlcmHillClimbing hillClimbing = new OlcmHillClimbing(400, 0.5, expansionOperators, simplificationOperators);
        return hillClimbing.learnModel(bn, data, dataName, em);
    }
}
