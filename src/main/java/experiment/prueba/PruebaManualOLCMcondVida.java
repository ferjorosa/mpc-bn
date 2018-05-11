package experiment.prueba;

import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.util.distance.Hellinger;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 10/05/2018.
 *
 * La idea es sustituir el modelo OLHC_real_condvida_1 por uno que mezcle un HLCM con arcos
 */
public class PruebaManualOLCMcondVida {

    public static void main(String[] args) throws Exception {
        /** 12 MVs real cond_vida data */
        String olcm_condVida_12_train_string = "experiments/real/data/cond_12.arff";
        DiscreteData olcm_condVida_12_train = DataFileLoader.loadDiscreteData(olcm_condVida_12_train_string);

        AbstractEM em = new ParallelEM(new EmConfig(), ScoreType.BIC);

        DiscreteBayesNet manual_model = createModel(olcm_condVida_12_train);

        LearningResult<DiscreteBayesNet> manual_model_result = em.learnModel(manual_model, olcm_condVida_12_train);
        System.out.println("Score: " + manual_model_result.getScoreValue());

        System.out.println("\n\n Average hellinger distances");
        List<Double> hellinger = Hellinger.averageClusterDistances(manual_model_result.getBayesianNetwork());

        for(Double d: hellinger)
            System.out.println(d + "\n");
    }

    private static DiscreteBayesNet createModel(DiscreteData data) {

        DiscreteVariable hs180 = data.getVariable("hs180").get();
        DiscreteVariable hs190 = data.getVariable("hs190").get();
        DiscreteVariable db100 = data.getVariable("db100").get();
        DiscreteVariable hs060 = data.getVariable("hs060").get();
        DiscreteVariable hs050 = data.getVariable("hs050").get();
        DiscreteVariable h79_u = data.getVariable("h79_u").get();
        DiscreteVariable hs040 = data.getVariable("hs040").get();
        DiscreteVariable hc190 = data.getVariable("hc190").get();
        DiscreteVariable family_size = data.getVariable("family_size").get();
        DiscreteVariable hh081 = data.getVariable("hh081").get();
        DiscreteVariable hs070 = data.getVariable("hs070").get();
        DiscreteVariable hs100 = data.getVariable("hs100").get();

        DiscreteVariable variable531 = new DiscreteVariable(2, VariableType.LATENT_VARIABLE, "variable531");
        DiscreteVariable variable119 = new DiscreteVariable(3, VariableType.LATENT_VARIABLE, "variable119");
        DiscreteVariable variable681 = new DiscreteVariable(2, VariableType.LATENT_VARIABLE, "variable681");

        List<DiscreteVariable> manifestVars = new ArrayList<>();
        manifestVars.add(hs180);manifestVars.add(hs190);manifestVars.add(db100);manifestVars.add(hs060);manifestVars.add(hs050);
        manifestVars.add(h79_u);manifestVars.add(hs040);manifestVars.add(hc190);manifestVars.add(family_size);manifestVars.add(hh081);
        manifestVars.add(hs070);manifestVars.add(hs100);

        DiscreteBayesNet bn = new DiscreteBayesNet();
        DiscreteBeliefNode variable531Node = bn.addNode(variable531);
        DiscreteBeliefNode variable119Node = bn.addNode(variable119);
        DiscreteBeliefNode variable681Node = bn.addNode(variable681);

        for(DiscreteVariable var : manifestVars)
            bn.addNode(var);

        // 531
        bn.addEdge(bn.getNode(hs180), variable531Node);
        bn.addEdge(bn.getNode(hs190), variable531Node);
        bn.addEdge(bn.getNode(db100), variable531Node);
        bn.addEdge(bn.getNode(hs060), variable531Node);

        // 119
        bn.addEdge(variable531Node, variable119Node);

        bn.addEdge(bn.getNode(hs050), variable119Node);
        bn.addEdge(bn.getNode(h79_u), variable119Node);
        bn.addEdge(bn.getNode(hs040), variable119Node);
        bn.addEdge(bn.getNode(hc190), variable119Node);

        // 681
        bn.addEdge(bn.getNode(family_size), variable681Node);
        bn.addEdge(bn.getNode(hh081), variable681Node);
        bn.addEdge(bn.getNode(hs070), variable681Node);
        bn.addEdge(bn.getNode(hs100), variable681Node);

        // family -> hc190
        bn.addEdge(bn.getNode(hc190), bn.getNode(family_size));

        return  bn;
    }

}
