package articulo.scripts.export;

import articulo.facet.DataFacet;
import voltric.variables.DiscreteVariable;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Metodo simple que genera un archivo CSV con una matriz de distancias
 */
public class ExportMatrixCSV {

    public static void export(String filePath, double[][] distanceMatrix) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filePath), "UTF-8"));

        // Escribimos la primera linea donde se exponen las columnas de la matriz
        for(int i=0; i < distanceMatrix.length; i++)
            if(i == 0)
                writer.print("\"\",\"c_"+i + "\"");
            else if(i == distanceMatrix.length - 1)
                writer.println(",\"c_" + i + "\"");
            else
                writer.print(",\"c_"+i + "\"");

        // Iteramos por las rows de la matriz de distancias para
        for(int i = 0; i <distanceMatrix.length; i++){
            // The first column of the row is always the corresponding cluster label
            writer.print("\"c_"+i+ "\",");
            for(int j = 0; j < distanceMatrix[i].length; j++){
                if(j == distanceMatrix[i].length - 1)
                    // The last column never ends with a comma
                    writer.println(distanceMatrix[i][j]);
                else
                    writer.print(distanceMatrix[i][j]+",");
            }
        }

        writer.close();
    }

    public static void export(String filePath, double[][] distanceMatrix, List<DataFacet> facets) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filePath), "UTF-8"));
        List<DiscreteVariable> firstVarOfFacet = facets.stream().map(x->x.getVariables().get(0)).collect(Collectors.toList());

        // Escribimos la primera linea donde se exponen las columnas de la matriz
        for(int i=0; i < distanceMatrix.length; i++)
            if(i == 0)
                writer.print("\"\",\""+firstVarOfFacet.get(i).getName()+ "\"");
            else if(i == distanceMatrix.length - 1)
                writer.println(",\"" + firstVarOfFacet.get(i).getName() + "\"");
            else
                writer.print(",\""+ firstVarOfFacet.get(i).getName() + "\"");

        // Iteramos por las rows de la matriz de distancias para
        for(int i = 0; i <distanceMatrix.length; i++){
            // The first column of the row is always the corresponding cluster label
            writer.print("\""+firstVarOfFacet.get(i).getName()+ "\",");
            for(int j = 0; j < distanceMatrix[i].length; j++){
                if(j == distanceMatrix[i].length - 1)
                    // The last column never ends with a comma
                    writer.println(distanceMatrix[i][j]);
                else
                    writer.print(distanceMatrix[i][j]+",");
            }
        }

        writer.close();
    }

}
