package articulo.learning;

import voltric.clustering.singleview.HiddenNaiveBayes;
import voltric.data.DiscreteData;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.model.DiscreteBayesNet;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;
import voltric.util.information.mi.NMI;
import voltric.util.information.mi.normalization.NMImax;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by equipo on 18/04/2018.
 *
 * Distingo 2 metodos privados:
 *
 *      - El de añadir cada variable al LCM ya existente para ver si mejora su Hellinger
 *      - El de formar un LCM entre las 2 variables con mayor NMI del vector actual
 *      
 * TODO: Nos quedamos con la MV que mejora la que mas la distancia de Hellinger entre los clusters de la LV
 */
public class HellingerFinder {


    private static double thresholdParaEM = 0.5;

    public static DiscreteBayesNet find(DiscreteData data, AbstractEM em) {

        /** Antes de comenzar, calculamos la NMI entre cada par de variables del dataSet */
        Map<DiscreteVariable, Map<DiscreteVariable, Double>> pairValues = NMI.computePairwise(data.getVariables(), data, new NMImax());

        List<DiscreteVariable> currentVariables = new ArrayList<>();
        currentVariables.addAll(data.getVariables());

        List<HLCM> currentLCMs = new ArrayList<>();

        /** Iteramos hasta que el numero de variables del vector se encuentre vacio */
        while(currentVariables.size() > 0) {

            // Si no existe ningun LCM, creamos un nuevo con el par de variables que tienen mejor valor NMI
            if(currentLCMs.size() == 0){
                HLCM best_lcm = formBestLCM(data, em, currentVariables, pairValues);
                currentLCMs.add(best_lcm);
            }
            // En cambio, si existen uno o mas LCMs, probamos cada una de las currentVariables para añadirlas al LCM que mejoren
            // Si no son capaces de mejorarlo, creamos un LCM con el par de variables cuyo valor de NMI es mas alto
            else {

            }
        }

        // Iteramos hasta que el conjunto de currentVariables se encuentre vacio

        // Si el conjunto de currentLCMs se enc


    }

    private static HLCM formBestLCM(DiscreteData data, AbstractEM em, List<DiscreteVariable> currentVariables, Map<DiscreteVariable, Map<DiscreteVariable, Double>> pairValues) {

        /** Despues seleccionamos el par de variables con mayor NMI para que formen una particion entre ambos */
        DiscreteVariable bestFirstVar = null;
        DiscreteVariable bestSecondVar = null;
        double bestNmiValue = -1;

        /** The keyset is filtered according to the vector of current variables */
        for(DiscreteVariable firstVar: pairValues.keySet().stream().filter(x -> !currentVariables.contains(x)).collect(Collectors.toList())){

            Map<DiscreteVariable, Double> secondVarsWithValues = pairValues.get(firstVar);

            for(DiscreteVariable secondVar: secondVarsWithValues.keySet()){
                double value = secondVarsWithValues.get(secondVar);
                if(value > bestNmiValue){
                    bestFirstVar = firstVar;
                    bestSecondVar = secondVar;
                    bestNmiValue = value;
                }
            }
        }

        /** Eliminamos las variables escogidas del vector de variables */
        List<DiscreteVariable> bestVariables = new ArrayList<>(2);
        bestVariables.add(bestFirstVar);
        bestVariables.add(bestSecondVar);

        currentVariables.removeAll(bestVariables);

        /** Formamos un LCM con las 2 mejores variables cuya cardinalidad se aprende gracias a un SEM*/
        HLCM lcm2Vars = (HLCM) HiddenNaiveBayes.learnModel(10, data.project(bestVariables), em, thresholdParaEM).getBayesianNetwork();

        return lcm2Vars;
    }

    // TODO: El currentVariables se le pasa como argumento porque vendria dado por el metodo superior
    /** Si devuelve null es que no hay ningun lcm que se haya visto mejorado por ninguna variable */
    private static void testVarWithLCMsHellinger(DiscreteData data, AbstractEM em, List<DiscreteVariable> currentVariables, List<HLCM> currentLCMs) {

        // Iteramos por el conjunto de currentVars y por el conjunto de currentPartitions
        for(DiscreteVariable var: currentVariables){
            for(HLCM lcm: currentLCMs) {

                // Añadimos la variable al LCM

                // Calculamos la distancia media de Hellinger entre sus clusters

                // Si la distancia media mejora, asociamos dicha variable con el HLCM que mas mejora su Hellinger distance
            }
        }

    }
}
