package articulo.learning.olcm.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.List;

/**
 * Created by equipo on 16/04/2018.
 */
public class IncreaseOlcmCard implements OlcmHcOperator{

    /** The maximum allowed cardinality value. */
    private int maxCardinality;

    /**
     * Main constructor.
     *
     * @param blackList The set of nodes that need to be avoided in the structure search process.
     * @param maxCardinality The maximum allowed cardinality value.
     */
    public IncreaseOlcmCard(List<Variable> blackList, int maxCardinality){
        this.maxCardinality = maxCardinality;
    }

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {

        // The BN is copied to avoid modifying current object.
        DiscreteBayesNet clonedNet = seedNet.clone();

        double bestModelScore = -Double.MAX_VALUE; // Log-likelihood related scores are negative
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        // Iteration through all the allowed BN nodes
        for(DiscreteVariable latentVar : clonedNet.getLatentVariables()) {

            // The cardinality of the LV must be lesser than the established maximum
            if(latentVar.getCardinality() < this.maxCardinality) {

                clonedNet = clonedNet.increaseCardinality(latentVar, 1);

                // After the LV has increased its cardinality, the resulting model is learned. If its score is improved, the LV is stored
                LearningResult<DiscreteBayesNet> increasedCardinalityResult = em.learnModel(clonedNet, data);
                if (increasedCardinalityResult.getScoreValue() > bestModelScore) {
                    bestModelScore = increasedCardinalityResult.getScoreValue();
                    bestModelResult = increasedCardinalityResult;
                }

                // The cardinality is reversed for the next iteration to have the initial BN.
                // Given that we have previously increased the LV's cardinality, it is a new LV, so we have to access it by its name.
                clonedNet = clonedNet.decreaseCardinality(clonedNet.getLatentVariable(latentVar.getName()), 1);
            }
        }

        DiscreteBayesNet resultBN = clonedNet;

        // If the model has been modified
        if(bestModelResult != null)
            return bestModelResult;

        return new LearningResult<>(null, bestModelScore, em.getScoreType());
    }
}
