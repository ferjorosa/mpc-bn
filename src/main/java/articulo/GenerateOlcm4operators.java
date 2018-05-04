package articulo;

import articulo.learning.HellingerFinder;
import articulo.learning.OlcmHillClimbing;
import articulo.learning.olcm.operator.*;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.bif.BnLearnBifFileWriter;
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
 * Created by equipo on 04/05/2018.
 */
public class GenerateOlcm4operators {

    public static void main(String[] args) throws Exception {
        /** 12MVs parkinson data */
        String olcm10MVs_train_string = "experiments/synthetic/12MVs/d897_filtered_10.arff";
        DiscreteData olcm10MVs_train = DataFileLoader.loadDiscreteData(olcm10MVs_train_string);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        DiscreteBayesNet initialOLCM = HellingerFinder.find(olcm10MVs_train, em);

        LearningResult<DiscreteBayesNet> olcm4op = apply4operators(initialOLCM, olcm10MVs_train, em);

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("olcm_parkinson_10.bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(olcm4op.getBayesianNetwork());
    }

    private static LearningResult<DiscreteBayesNet> apply4operators(DiscreteBayesNet bn, DiscreteData data, AbstractEM em) throws Exception {

        /** Expansion Operators */
        Set<OlcmHcOperator> expansionOperators = new HashSet<>();
        expansionOperators.add(new AddOlcmArc());
        //expansionOperators.add(new NewAddOlcmNode());
        expansionOperators.add(new IncreaseOlcmCard(10));

        /** Simplification Operators */
        Set<OlcmHcOperator> simplificationOperators = new HashSet<>();
        simplificationOperators.add(new RemoveOlcmArc());
        //simplificationOperators.add(new NewRemoveOlcmNode());
        simplificationOperators.add(new DecreaseOlcmCard(2));

        OlcmHillClimbing hillClimbing = new OlcmHillClimbing(400, 0.5, expansionOperators, simplificationOperators);

        return hillClimbing.learnModel(bn, data, em);
    }
}
