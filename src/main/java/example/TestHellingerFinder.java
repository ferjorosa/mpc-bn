package example;

import articulo.learning.HellingerFinder;

import articulo.learning.OlcmHillClimbing;
import articulo.learning.olcm.operator.*;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.ScoreType;
import voltric.learning.structure.hillclimbing.operator.HcOperator;
import voltric.model.DiscreteBayesNet;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by equipo on 19/04/2018.
 */
public class TestHellingerFinder {

    public static void main(String[] args) throws Exception {
        // Cargamos los datos

        String olcm_8MVs_1000 = "estudios/synthetic/8MVs/data/olcm8MVs_1000.arff";
        String d897_motor = "estudios/parkinson/data/d897_motor25.arff";
        String d897_nms = "estudios/parkinson/data/d897_nms30_s6.arff";

        String dataString = olcm_8MVs_1000;
        String dataName = "olcm_8MVs_1000";

        /** Load Weka data */
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(dataString));
        Instances dataWeka = loader.getDataSet();
        dataWeka.setClassIndex(0);

        /** Load Voltric data */
        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData(dataString);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        DiscreteBayesNet initialOLCM = HellingerFinder.find(dataVoltric, em);
        LearningResult<DiscreteBayesNet> olcmResult = apply6operators(initialOLCM, dataVoltric, em);
        System.out.println(olcmResult.getBayesianNetwork().toString());
        System.out.println(olcmResult.getScoreValue());
    }

    private static LearningResult<DiscreteBayesNet> apply6operators(DiscreteBayesNet bn, DiscreteData data, AbstractEM em) throws Exception {


        /** Expansion Operators */
        Set<OlcmHcOperator> expansionOperators = new HashSet<>();
        expansionOperators.add(new AddOlcmArc());
        expansionOperators.add(new AddOlcmNode());
        expansionOperators.add(new IncreaseOlcmCard(10));

        /** Simplification Operators */
        Set<OlcmHcOperator> simplificationOperators = new HashSet<>();
        simplificationOperators.add(new RemoveOlcmArc());
        simplificationOperators.add(new RemoveOlcmNode());
        simplificationOperators.add(new DecreaseOlcmCard(2));
        
        OlcmHillClimbing hillClimbing = new OlcmHillClimbing(400, 0.5, expansionOperators, simplificationOperators);

        return hillClimbing.learnModel(bn, data, "synthetic/8MVs", em);
    }
}
