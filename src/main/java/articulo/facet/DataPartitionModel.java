package articulo.facet;

import voltric.learning.LearningResult;
import voltric.model.DiscreteBayesNet;

/**
 * Created by equipo on 18/12/2017.
 */
public class DataPartitionModel {

    private LearningResult<DiscreteBayesNet> bnResult;

    private double brierScore;

    public DataPartitionModel(LearningResult<DiscreteBayesNet> bnResult, double brierScore) {
        this.bnResult = bnResult;
        this.brierScore = brierScore;
    }

    public LearningResult<DiscreteBayesNet> getBnResult() {
        return bnResult;
    }

    public double getBrierScore() {
        return brierScore;
    }
}
