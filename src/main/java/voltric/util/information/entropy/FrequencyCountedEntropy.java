package voltric.util.information.entropy;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by fernando on 15/04/17.
 */
public class FrequencyCountedEntropy {


    /**
     * Returns the entropy of the specified discrete dataSet.
     *
     * @param data the discrete dataSet.
     * @return the entropy value.
     */
    public static double compute(DiscreteData data){
        double entropy = 0.0;
        for(DiscreteDataInstance instance: data.getInstances()){
            double instanceProbability = data.getWeight(instance) / data.getTotalWeight();
            entropy += instanceProbability * Math.log(instanceProbability);
        }
        return -entropy;
    }

    public static double computeConditional(DiscreteVariable condVar, DiscreteData data){
        List<DiscreteVariable> condVars = new ArrayList<>();
        condVars.add(condVar);

        return FrequencyCountedEntropy.computeConditional(condVars, data);
    }

    // H(X|Y) = \sum p(x,y) log( p(x,y) / p(x) )
    public static double computeConditional(List<DiscreteVariable> condVars, DiscreteData data){

        if(condVars.isEmpty())
            return FrequencyCountedEntropy.compute(data);

        if(!data.getVariables().containsAll(condVars))
            throw new IllegalArgumentException("All the conditional variables must belong to the DataSet");

        DiscreteData yProjectedDataSet = data.project(condVars);

        double condEntropy = 0.0;

        for(DiscreteDataInstance instance: data.getInstances()){
            double Pxy = data.getFrequency(instance);
            double Py = yProjectedDataSet.getFrequency(instance);
            condEntropy += Pxy * Math.log(Pxy / Py);
        }

        return -condEntropy;
    }

    /**
     * Returns the sum of the individual entropies that belong to each data's variable.
     *
     * @param data the discrete dataSet.
     * @return the sum of individual entropies.
     */
    public static double computeSumOfIndividualEntropies(DiscreteData data){
        double sumOfEntropies = 0.0;
        for(DiscreteVariable variable: data.getVariables())
            // Computes the entropy of the data being projected to the variable's dimension and adds it to the sum
            sumOfEntropies += FrequencyCountedEntropy.compute(data.project(variable));
        return sumOfEntropies;
    }

    /**
     * Returns the entropy value for a specific probability.
     *
     * @param probability the probability value.
     * @return the partial entropy value.
     */
    public static double computePartialValue(double probability){
        return probability * Math.log(probability);
    }

}
