package example;

import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 18/04/2018.
 *
 * El proceso es simple, aprendemos el modelo o modelos que queramos, ya sea con EM, SEM full-OLCM
 * Despues comparamos el score/scores con los obtenidos por otros metodos como son el BI; EAST; MIM deAsbeh & Lerner
 */
public class OlcmVsHlcmInSynthetic {

    public static void main(String[] args) throws Exception {

        // Cargamos los datos

        String olcm_8MVs_1000 = "estudios/synthetic/8MVs/data/olcm8MVs_1000.arff";
        String olcm_8MVs_5000 = "estudios/synthetic/8MVs/data/olcm8MVs_5000.arff";
        String olcm_8MVs_10000 = "estudios/synthetic/8MVs/data/olcm8MVs_10000.arff";

        String dataString = olcm_8MVs_1000;
        String dataName = "olcm_8MVs_1000";

        /** Load Weka data */
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(dataString));
        Instances dataWeka = loader.getDataSet();
        dataWeka.setClassIndex(0);

        /** Load Voltric data */
        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData(dataString);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        DiscreteBayesNet olcmPerfectStruct = construct8MVs2LVs(dataVoltric);
        LearningResult<DiscreteBayesNet> olcmPerfectStructResult = em.learnModel(olcmPerfectStruct, dataVoltric);

        System.out.println("Score BIC: " + olcmPerfectStructResult.getScoreValue());
    }

    private static DiscreteBayesNet construct8MVs2LVs(DiscreteData data) {

        DiscreteVariable mv1 = data.getVariable("mv1").get();
        DiscreteVariable mv2 = data.getVariable("mv2").get();
        DiscreteVariable mv3 = data.getVariable("mv3").get();
        DiscreteVariable mv4 = data.getVariable("mv4").get();
        DiscreteVariable mv5 = data.getVariable("mv5").get();
        DiscreteVariable mv6 = data.getVariable("mv6").get();
        DiscreteVariable mv7 = data.getVariable("mv7").get();
        DiscreteVariable mv8 = data.getVariable("mv8").get();

        DiscreteVariable firstClustVar = new DiscreteVariable(3, VariableType.LATENT_VARIABLE, "c1");
        DiscreteVariable secondClustVar = new DiscreteVariable(2, VariableType.LATENT_VARIABLE, "c2");

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

        return bn;
    }
}
