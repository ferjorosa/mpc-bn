package articulo.facet;

import util.SimpleHashCreator;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 11/12/2017.
 */
public class DataFacet {

    private List<DiscreteVariable> variables;

    private DiscreteBayesNet bayesNet;

    private double entropy;

    private double totalCorrelation;

    private int index;

    private int myHashCode;

    public DataFacet(List<DiscreteVariable> nonRepeatedVariables, double entropy, double totalCorrelation, int index){
        this.variables = nonRepeatedVariables;
        this.entropy = entropy;
        this.totalCorrelation = totalCorrelation;
        this.index = index;

        // Create an empty BN (no edges)
        this.bayesNet = new DiscreteBayesNet("DataFacet BN("+index+")");
        for(DiscreteVariable var: nonRepeatedVariables)
            bayesNet.addNode(var);

        this.myHashCode = (int) SimpleHashCreator.createHash(nonRepeatedVariables);
    }

    public DataFacet(List<DiscreteVariable> nonRepeatedVariables, DiscreteBayesNet bayesNet, double entropy, double totalCorrelation, int index){
        this.variables = nonRepeatedVariables;
        this.bayesNet = bayesNet;
        this.entropy = entropy;
        this.totalCorrelation = totalCorrelation;
        this.index = index;
        this.myHashCode = (int) SimpleHashCreator.createHash(nonRepeatedVariables);
    }

    public String getName() {
        List<String> variablesNames = this.variables.stream().map(x->x.getName()).collect(Collectors.toList());
        String variableNamesString = "";
        for(String varName: variablesNames)
            variableNamesString += varName + " ;";
        return variableNamesString;
    }

    public List<DiscreteVariable> getVariables() {
        return variables;
    }

    public DiscreteBayesNet getBayesNet() {
        return bayesNet;
    }

    public double getEntropy() {
        return entropy;
    }

    public double getTotalCorrelation() {
        return totalCorrelation;
    }

    public int getIndex() {
        return index;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DataFacet dataFacet = (DataFacet) o;

        return dataFacet.hashCode() == this.hashCode();
    }

    @Override
    public int hashCode() {
        return myHashCode;
    }
}
