package example;

import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.util.information.entropy.BnFactorizationEntropy;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

/**
 * Created by equipo on 10/04/2018.
 *
 * Voy a probar si es posible tener H(X,Y,Z) < H(X,Y) cuando la entropia se encuentra factorizada con una BN
 *
 * Esta mal, asi no se obtiene la entropia factorizada de una BN, es necesario utilizar factorizacion + inferencia para obtener
 * las JPDs de las variables en cuestion
 *
 * TODO: Nota 17-04-2018: La entropia factorizada no se calcula de esta manera, por eso da error
 */
public class FactorizedEntropyProblem {

    public static void main(String[] args) throws Exception {

        // Tengo que crear la BN a mano y ajustarle las probabilidades para que se cumpla lo que busco
        DiscreteBayesNet bn = new DiscreteBayesNet("testEntropy");

        DiscreteBeliefNode nodeA = bn.addNode(new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "A"));
        DiscreteBeliefNode nodeB = bn.addNode(new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "B"));
        DiscreteBeliefNode nodeC = bn.addNode(new DiscreteVariable(2, VariableType.MANIFEST_VARIABLE, "C"));

        bn.addEdge(nodeA, nodeC);
        bn.addEdge(nodeB, nodeC);

        // TODO: Recordar cambiar a private el metodo de createFunction
        Function fCreator = Function.createIdentityFunction();

        DiscreteVariable c[] = new DiscreteVariable[1];
        c[0] = nodeC.getVariable();
        double[] cCells = {0.98, 0.02};
        int[] cMagnitudes = {1};
        //Function cptC = fCreator.createFunction(c, cCells, cMagnitudes);

        DiscreteVariable ac[] = new DiscreteVariable[2];
        ac[0]= nodeA.getVariable();
        ac[1] = nodeC.getVariable();
        double[] acCells = {0.4,0.3,0.2,0.1};
        int[] acMagnitudes = {2,1};
        //Function cptAC = fCreator.createFunction(ac, acCells, acMagnitudes);

        DiscreteVariable bc[] = new DiscreteVariable[2];
        bc[0]= nodeB.getVariable();
        bc[1] = nodeC.getVariable();
        double[] bcCells = {0.8,0.05,0.1,0.05};
        int[] bcMagnitudes = {2,1};
        //Function cptBC = fCreator.createFunction(bc, bcCells, bcMagnitudes);


        //bn.getNode("A").setCpt(cptAC);
        //bn.getNode("B").setCpt(cptBC);
        //bn.getNode("C").setCpt(cptC);

        System.out.println(bn);

        System.out.println(BnFactorizationEntropy.compute(bn));

        CliqueTreePropagation cliqueTreePropagation = new CliqueTreePropagation(bn);
        cliqueTreePropagation.propagate();

        Function posteriorA = cliqueTreePropagation.computeBelief(nodeA.getVariable());
        Function posteriorB = cliqueTreePropagation.computeBelief(nodeB.getVariable());
        Function posteriorC = cliqueTreePropagation.computeBelief(nodeC.getVariable());
    }
}
