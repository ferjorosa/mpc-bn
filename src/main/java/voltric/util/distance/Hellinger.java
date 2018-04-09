package voltric.util.distance;

import voltric.model.DiscreteBayesNet;

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
}
