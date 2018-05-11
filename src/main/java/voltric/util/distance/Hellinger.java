package voltric.util.distance;

import voltric.model.DiscreteBayesNet;

import java.util.ArrayList;
import java.util.List;

/**
 * La distancia de Hellinger se encuentra fuertemente relacionada con la distancia de Bhattacharyya
 * H^{2}(P,Q) = 1 - BC(P,Q) => H(P,Q) = sqrt(1 - BC(P,Q))
 */
public class Hellinger {


    // Cuando son HLCMs las hijas de una LV pueden ser del tipo manifest o latent, para calcular la distancia
    public static List<double[][]> clusterDistances(DiscreteBayesNet mcm) {

        List<double[][]> bhattacharyyaDistances = Bhattacharyya.clusterDistances(mcm);
        List<double[][]> hellingerDistances = new ArrayList<>(bhattacharyyaDistances.size());

        for(double[][] bhattacharyyaMatrix: bhattacharyyaDistances){
            double[][] distanceMatrix = new double[bhattacharyyaMatrix.length][bhattacharyyaMatrix.length];

            for(int i = 0; i< bhattacharyyaMatrix.length; i++)
                for(int j = 0; j < bhattacharyyaMatrix.length; j++)
                    distanceMatrix[i][j] = 0;

            for(int i = 0; i < bhattacharyyaMatrix.length; i++)
                for(int j = 0; j < bhattacharyyaMatrix.length; j++)
                    if(i != j)
                        distanceMatrix[i][j] = Math.sqrt(1 - bhattacharyyaMatrix[i][j]);

            hellingerDistances.add(distanceMatrix);
        }
        return hellingerDistances;
    }

    public static List<Double> averageClusterDistances(DiscreteBayesNet mcm) {

        List<double[][]> hellingerMatrixes = clusterDistances(mcm);

        List<Double> partitionHellingerValues = new ArrayList<>();

        double sumHellingerValues = 0;

        for(double[][] hellingerMatrix: hellingerMatrixes){
            sumHellingerValues = 0;
            List<Double> hellingerValues = new ArrayList<>();

            // Iteramos por la upper triangular matrix sin contar la diagonal que es 0
            for (int i = 0; i < hellingerMatrix.length; i++){
                for (int j = i; j < hellingerMatrix.length; j++)
                    if (j != i)
                        hellingerValues.add(hellingerMatrix[i][j]);
            }

            for(double value: hellingerValues)
                sumHellingerValues += value;

            // Añadimos la distancia media de Hellinger para la particion en cuestion
            partitionHellingerValues.add(sumHellingerValues / hellingerValues.size());
        }

        return partitionHellingerValues;
    }

    /** Calcula la matriz de distancias de Hellinger cuando el input es un LCM */
    public static double[][] clusterDistancesLCM(DiscreteBayesNet lcm) {
        double[][] bhattacharyyaDistances = Bhattacharyya.clusterDistancesLCM(lcm);

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

    public static double averageClusterDistancesLCM(DiscreteBayesNet lcm) {
        double[][] hellingerDistances = clusterDistancesLCM(lcm);

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
