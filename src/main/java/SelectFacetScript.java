/**
 * Created by equipo on 09/04/2018.
 * TODO: Ojo con las facetas descartas. El tamaño del vector "groups" es igual al numero de facets que puede ser inferior al de variables en el dataset
 */
public class SelectFacetScript {

    public static void main(String[] args) throws Exception {

        // Primero copiamos los grupos asignados a cada variable
        int[] groups = {1,1,1,1,1,2,3,1,1,4,4,1,3,1,3,1,3,3,3,1,3,5,1,1,3};

        // Despues calculamos la distancia media de Hellinger para el LCM asociado a cada una de las facetas

        // Escogemos de cada grupo de facetas aquella que tiene una distancia media de Hellinger mas alta

        // Las almacenamos ya que formaran la estructura base del OLCM final
    }
}
