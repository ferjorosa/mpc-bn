package articulo.scripts.util;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.io.data.DataFileLoader;
import voltric.variables.DiscreteVariable;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by equipo on 14/01/2018.
 */
public class GenerateDataEAST {

    public static void main(String[] args) throws IOException{
        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData("estudios/synthetic/8MVs/data/olcm8MVs_5000.arff");

        GenerateDataEAST.generate(dataVoltric, "estudios/synthetic/8MVs/data/olcm8MVs_5000.data");
    }

    public static void generate(DiscreteData data, String filePathString) throws IOException{

        FileWriter fw = new FileWriter(filePathString);

        // write data name
        fw.write("Name: "+ data.getName());

        // write attributes
        fw.write("\n\n//Variables: name of variable followed by names of states\n");

        for(DiscreteVariable var: data.getVariables()) {
            fw.write("\n" + var.getName()+": ");
            for(String state: var.getStates())
                fw.write(state+" ");
        }

        fw.write("\n\n//Records: Numbers in the last column are frequencies.\n\n");
        // write instances
        for (DiscreteDataInstance instance : data.getInstances())
            writeInstanceToFile(instance, fw, " ");

        fw.close();
    }

    private static void writeInstanceToFile(DiscreteDataInstance instance, FileWriter writer, String separator) throws IOException{
        String instanceString = instanceToString(instance, separator);
        double weight =  instance.getData().getWeight(instance);

        writer.write(instanceString + "   "+ weight +"\n");
    }

    private static String instanceToString(DiscreteDataInstance instance, String separator) {
        String s = "";

        // Append all the columns of the DataInstance with  the separator except the last one
        for(int i = 0; i < instance.getTextualValues().size() - 1; i++)
            s += instance.getNumericValue(i) + separator;
        // Append the last column of the instance without the separator
        s += instance.getNumericValue(instance.getTextualValues().size() - 1);
        return s;
    }

    /*
    private static String instanceToString(DiscreteDataInstance instance, String separator) {
        String s = "";

        // Append all the columns of the DataInstance with  the separator except the last one
        for(int i = 0; i < instance.getTextualValues().size() - 1; i++)
            s += instance.getTextualValue(i) + separator;
        // Append the last column of the instance without the separator
        s += instance.getTextualValue(instance.getTextualValues().size() - 1);
        return s;
    }
    */
}
