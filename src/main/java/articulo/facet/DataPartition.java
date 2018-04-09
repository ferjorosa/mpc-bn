package articulo.facet;

import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 11/12/2017.
 */
public class DataPartition {

    private List<DataFacet> facets;

    private double entropy;

    private double totalCorrelation;

    private List<DiscreteVariable> variables;

    public DataPartition(DataFacet facet) {
        this.facets = new ArrayList<>();
        this.facets.add(facet);
        this.entropy = facet.getEntropy();
        this.totalCorrelation = facet.getTotalCorrelation();
        this.variables = facet.getVariables();
    }

    public DataPartition(List<DataFacet> facets, List<DiscreteVariable> nonRepeatedVariables, double entropy, double totalCorrelation){
        this.facets = facets;
        this.variables = nonRepeatedVariables;
        this.entropy = entropy;
        this.totalCorrelation = totalCorrelation;
    }

    public List<DataFacet> getFacets() {
        return facets;
    }

    public double getEntropy() {
        return entropy;
    }

    public double getTotalCorrelation() {
        return totalCorrelation;
    }

    public List<DiscreteVariable> getVariables() {
        return variables;
    }

    public boolean contains(DataFacet facet){
        return this.facets.contains(facet);
    }
}
