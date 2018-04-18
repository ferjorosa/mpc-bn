package articulo.scripts.syntheticbn;

import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.bif.OldBifFileWriter;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 17/04/2018.
 */
public class DefinedOLCM {

    public static void main(String[] args) throws Exception {
        olcm8MVs2LVs();
    }

    /**
     * Definimos manualmente con sus probabilidades un OLCM de 2 LVs (con 3 y 2 estados respectivamente) y 8 MVs,
     * la cual ha sido generada en AMIDST
     */
    private static void olcm8MVs2LVs() throws Exception{
        // Definimos la estructura unicamente para saber como debemos escribir el archivo BIF de forma consistente

        DiscreteVariable mv1 = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "mv1");
        DiscreteVariable mv2 = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "mv2");
        DiscreteVariable mv3 = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "mv3");
        DiscreteVariable mv4 = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "mv4");
        DiscreteVariable mv5 = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "mv5");
        DiscreteVariable mv6 = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "mv6");
        DiscreteVariable mv7 = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "mv7");
        DiscreteVariable mv8 = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "mv8");

        DiscreteVariable firstClustVar = new DiscreteVariable(3, VariableType.MANIFEST_VARIABLE, "c1");
        DiscreteVariable secondClustVar = new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "c2");

        List<DiscreteVariable> manifestVars = new ArrayList<>();
        manifestVars.add(mv1);manifestVars.add(mv2);manifestVars.add(mv3);manifestVars.add(mv4);manifestVars.add(mv5);manifestVars.add(mv6);
        manifestVars.add(mv7);manifestVars.add(mv8);

        DiscreteBayesNet bn = new DiscreteBayesNet();
        DiscreteBeliefNode firstClustNode = bn.addNode(firstClustVar);
        DiscreteBeliefNode secondClustNode = bn.addNode(secondClustVar);

        for(DiscreteVariable var : manifestVars)
            bn.addNode(var);

        bn.addEdge(bn.getNode(mv1), firstClustNode);
        bn.addEdge(bn.getNode(mv2), firstClustNode);
        bn.addEdge(bn.getNode(mv3), firstClustNode);
        bn.addEdge(bn.getNode(mv4), firstClustNode);
        bn.addEdge(bn.getNode(mv5), firstClustNode);
        bn.addEdge(bn.getNode(mv6), firstClustNode);

        bn.addEdge(bn.getNode(mv5), secondClustNode);
        bn.addEdge(bn.getNode(mv6), secondClustNode);
        bn.addEdge(bn.getNode(mv7), secondClustNode);
        bn.addEdge(bn.getNode(mv8), secondClustNode);

        bn.randomlyParameterize();

        System.out.println(bn);

        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream("olcm_8MVs.bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(bn);

        // Export model in OBIF format
        OldBifFileWriter.writeBif("olcm_8MVs.obif", bn);
    }
}
