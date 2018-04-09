package articulo;

import voltric.graph.Edge;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 11/01/2018.
 */
public class MarkovBlanketGenerator {

    /**
     * Iteramos por cada uno de los nodos de la red Bayesiana y devolvemos el nodo + su Markov Blanket
     *
     * @param bn la red Bayesiana en cuestion
     * @return
     */
    public static List<DiscreteBayesNet> generate(DiscreteBayesNet bn) {

        List<DiscreteBayesNet> markovBlankets = new ArrayList<>();

        for(DiscreteBeliefNode node: bn.getNodes()){
            DiscreteBayesNet mb = new DiscreteBayesNet();
            // Añadimos el nodo en el MB, el cual sera utilizado en la generacion de arcos dentro del MB
            DiscreteBeliefNode mbNode = mb.addNode(node.getVariable());

            // 1 - Añadimos sus padres al MB junto con sus correspondientes arcos desde ellos a 'node'
            for(Edge<Variable> parentEdge: node.getParentEdges()){
                DiscreteBeliefNode mbParent = mb.addNode((DiscreteVariable) parentEdge.getTail().getContent());
                mb.addEdge(mbNode, mbParent); // From parent to node
            }

            List<DiscreteBeliefNode> mbChildren = new ArrayList<>();

            // 2 - Añadimos sus hijos al MB junto con sus correspondientes arcos desde 'node' a ellos
            for(Edge<Variable> childEdge: node.getChildEdges()){
                DiscreteBeliefNode mbChild = mb.addNode((DiscreteVariable) childEdge.getHead().getContent());
                mb.addEdge(mbChild, mbNode); // From node to child
                mbChildren.add(mbChild);
            }

            // 3- Una vez añadidos todos ellos, añadimos los padres de sus hijos que no se encuentren ya en el MB
            for(DiscreteBeliefNode mbChild: mbChildren) {

                List<DiscreteVariable> otherParents = bn.getNode(mbChild.getVariable()).getDiscreteParentVariables()
                        .stream()
                        .filter(x-> !x.equals(mbNode.getVariable()))
                        .filter(x-> !mb.containsVar(x))
                        .collect(Collectors.toList());

                // Una vez seleccionados los padres, los añadimos junto con sus respectivos arcos
                for(DiscreteVariable otherParent: otherParents){
                    DiscreteBeliefNode otherParentNode = mb.addNode(otherParent);
                    mb.addEdge(mbChild, otherParentNode);
                }
            }

            markovBlankets.add(mb);
        }

        return markovBlankets;
    }

    // Version del MB que no incluye a los padres del hijo
    public static List<DiscreteBayesNet> generateSimple(DiscreteBayesNet bn) {

        List<DiscreteBayesNet> markovBlankets = new ArrayList<>();

        for(DiscreteBeliefNode node: bn.getNodes()){
            DiscreteBayesNet mb = new DiscreteBayesNet();
            // Añadimos el nodo en el MB, el cual sera utilizado en la generacion de arcos dentro del MB
            DiscreteBeliefNode mbNode = mb.addNode(node.getVariable());

            // 1 - Añadimos sus padres al MB junto con sus correspondientes arcos desde ellos a 'node'
            for(Edge<Variable> parentEdge: node.getParentEdges()){
                DiscreteBeliefNode parent = mb.addNode((DiscreteVariable) parentEdge.getTail().getContent());
                mb.addEdge(mbNode, parent); // From parent to node
            }

            // 2 - Añadimos sus hijos al MB junto con sus correspondientes arcos desde 'node' a ellos
            for(Edge<Variable> childEdge: node.getChildEdges()){
                DiscreteBeliefNode child = mb.addNode((DiscreteVariable) childEdge.getHead().getContent());
                mb.addEdge(child, mbNode); // From node to child
            }

            markovBlankets.add(mb);
        }

        return markovBlankets;
    }
}
