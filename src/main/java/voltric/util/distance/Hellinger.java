package voltric.util.distance;

import voltric.model.DiscreteBayesNet;

import java.util.ArrayList;
import java.util.List;

/**
 * La distancia de Hellinger se encuentra fuertemente relacionada con la distancia de Bhattacharyya
 * H^{2}(P,Q) = 1 - BC(P,Q) => H(P,Q) = sqrt(1 - BC(P,Q))
 */
public class Hellinger {


    public static double[][] clusterDistances(DiscreteBayesNet lcm) {
        double[][] bhattacharyyaDistances = Bhattacharyya.clusterDistances(lcm);

        double[][] distanceMatrix = new double[bhattacharyyaDistances.length][bhattacharyyaDistances.length];
        for(int i = 0; i< bhattacharyyaDistances.length; i++)
            for(int j = 0; j < bhattacharyyaDistances.length; j++)
                distanceMatrix[i][j] = 0;

        for(int i = 0; i < bhattacharyyaDistances.length; i++)
            for(int j = 0; j < bhattacharyyaDistances.length; j++)
                if(i != j)
                    distanceMatrix[i][j] = Math.sqrt(1 - bhattacharyyaDistances[i][j]);

        return distanceMatrix;
    }

    public static double averageClusterDistances(DiscreteBayesNet lcm) {
        double[][] hellingerDistances = clusterDistances(lcm);

        List<Double> hellingerValues = new ArrayList<>();

        // Iteramos por la upper triangular matrix sin contar la diagonal que es 0
        for (int i = 0; i < hellingerDistances.length; i++){
            for (int j = i; j < hellingerDistances.length; j++)
                if (j != i)
                    hellingerValues.add(hellingerDistances[i][j]);
        }

        double sumHellingerValues = 0;
        for(double value: hellingerValues)
            sumHellingerValues += value;

        return sumHellingerValues / hellingerValues.size();
    }
}
