package articulo;

import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;
import weka.bif.XmlBifReader;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 15/01/2018.
 */
public class Codigo {

    public Instances projectDataWeka(List<DiscreteVariable> variables, Instances dataWeka) throws Exception{
        List<String> projectionVariablesNames = variables.stream().map(DiscreteVariable::getName).collect(Collectors.toList());

        /** Definimos los parametros de filter utilizado */
        String projectionVarNamesString = "";
        for(String varName: projectionVariablesNames)
            projectionVarNamesString += varName + "|";

        String optionsString = "-E ^(?!("+projectionVarNamesString+")$).*$";
        String[] options = weka.core.Utils.splitOptions(optionsString);

        /** Eliminamos los atributos de dataWeka con weka.RemoveByName*/
        RemoveByName removeByNameFilter = new RemoveByName();
        removeByNameFilter.setOptions(options);
        removeByNameFilter.setInputFormat(dataWeka);
        Instances projectedDataWeka = Filter.useFilter(dataWeka, removeByNameFilter);
        projectedDataWeka.setClassIndex(0);

        return projectedDataWeka;
    }

    public DiscreteBayesNet learnBnUsingWeka(List<DiscreteVariable> variables, Instances dataWeka) throws Exception{

        // Project dataWeka to the 'variables' space
        Instances projectedDataWeka = projectDataWeka(variables, dataWeka);

        BayesNet emptyNet = new BayesNet();
        HillClimber oSearchAlgorithm = new HillClimber();
        oSearchAlgorithm.setInitAsNaiveBayes(false);
        oSearchAlgorithm.setMaxNrOfParents(15);
        emptyNet.setSearchAlgorithm(oSearchAlgorithm);
        emptyNet.buildClassifier(projectedDataWeka); // Learns a network from the empty net
        String bifNet = emptyNet.toXMLBIF03();

        return XmlBifReader.processString(bifNet);
    }
}
