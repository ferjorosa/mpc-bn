package voltric.util.distance;

import org.apache.commons.math3.util.Combinations;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.util.*;

/**
 * Calcula la distancia de Bhattacharyya entre dos listas de Function correspondientes a las CPTs dadas las los valores de la clase
 *
 * EJ: Comparar las distribuciones multinomiales de un Naïve Bayes dado que C=1 y C=2
 */
public class Bhattacharyya {

    public static double[][] clusterDistances(DiscreteBayesNet lcm) {
        // TODO: Faltaria una comprobacion mas fuerte de que se trata de un LCM
        if(lcm.getLatentVariables().size() != 1)
            throw new IllegalArgumentException("The BN has to be an LCM");

        // Obtenemos la raiz del modelo Naïve Bayes
        DiscreteVariable root = lcm.getLatentVariables().get(0);

        // Inicializamos la matriz de distancias con 0s
        double[][] distanceMatrix = new double[root.getCardinality()][root.getCardinality()];
        for(int i = 0; i<root.getCardinality(); i++)
            for(int j = 0; j < root.getCardinality(); j++)
                distanceMatrix[i][j] = 0;

        // Creates a CliqueTreePropagation instance to do the inference
        CliqueTreePropagation inferenceEngine = new CliqueTreePropagation(lcm);

        Function[][] localCPTs = new Function[root.getCardinality()][lcm.getManifestVariables().size()];

        Map<DiscreteVariable, Integer> evidence = new HashMap<>();
        for(int i = 0; i < root.getCardinality(); i++) {
            // Asignamos la evidencia con el valor de la variable de cluster
            evidence.put(root, i);
            inferenceEngine.setEvidence(evidence);

            // Propagamos la evidencia
            inferenceEngine.propagate();

            // Recogemos las CPTs marginales correspondientes a cada variable Manifest del modelo
            for (int j = 0; j < lcm.getManifestVariables().size(); j++)
                localCPTs[i][j] = inferenceEngine.computeBelief(lcm.getManifestVariables().get(j));
        }

        // Using Apache Math, create a set of index combinations of cluster pairs
        Iterator<int[]> clusterPairCombinations = new Combinations(root.getCardinality(), 2).iterator();

        // Calculamos las distancias entre cada par de combinaciones
        while(clusterPairCombinations.hasNext()){
            // Indices de los clusters a comparar
            int[] combination = clusterPairCombinations.next();
            int c1 = combination[0]; // Index of the first cluster to compare
            int c2 = combination[1]; // Index of the second cluster to compare

            // Dado que tratamos con un modelo Naïve Bayes, la distribucion de probabilidad se factoriza como un producto de distribuciones marginales
            // Iteramos por la matriz de distribuciones marginales (localCPTs)

            // Esta distancia se calcula como el producto de las distancias marginales (ver formula)
            double clusterDistance = 1;
            for(int i = 0; i< lcm.getManifestVariables().size(); i++)
                clusterDistance *= distance(localCPTs[c1][i], localCPTs[c2][i]);

            distanceMatrix[c1][c2] = clusterDistance;
            distanceMatrix[c2][c1] = clusterDistance; // Symmetric
        }
        return distanceMatrix;
    }

    public static double distance(Function a, Function b) {
        if(a.getDimension() != b.getDimension() || a.getDomainSize() != b.getDomainSize())
            throw new IllegalArgumentException("Both functions need to have the same number of variables and states");

        double distance = 0;
        for(int i = 0; i < a.getDomainSize(); i++)
            distance += Math.sqrt(a.getCells()[i] * b.getCells()[i]);

        return distance;
    }
}
