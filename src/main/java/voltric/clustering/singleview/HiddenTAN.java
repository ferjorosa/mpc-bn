package voltric.clustering.singleview;

import voltric.data.DiscreteData;
import voltric.graph.AbstractNode;
import voltric.graph.DirectedAcyclicGraph;
import voltric.graph.DirectedNode;
import voltric.graph.Edge;
import voltric.graph.weighted.WeightedUndirectedGraph;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.structure.chowliu.ChowLiu;
import voltric.model.DiscreteBayesNet;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;
import voltric.util.stattest.discrete.DiscreteStatisticalTest;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * El proceso es simple, se parte de un Hidden TAN de cardinalidad 2, se aumenta la cardinalidad y se recalcula el TAN,
 * si mejora el score se mantiene, en caso contrario se deja la cardinalidad
 *
 * TODO: El Chow-Liu debe permitir escoger que tipo de Mutual information quieres utilizar
 */
public class HiddenTAN {


    public static LearningResult<DiscreteBayesNet> learnModel(int maxCardinality,
                                                              DiscreteData dataSet,
                                                              DiscreteParameterLearning parameterLearning,
                                                              double threshold,
                                                              boolean randomChowLiuRoot,
                                                              DiscreteStatisticalTest statisticalTest){

        int currentCardinality = 2;
        LearningResult<DiscreteBayesNet> bestModel = learnHiddenTAN(currentCardinality, parameterLearning,dataSet,randomChowLiuRoot, statisticalTest);

        while(currentCardinality <= maxCardinality){

            // cardinality is increased
            currentCardinality = currentCardinality + 1;

            LearningResult<DiscreteBayesNet> currentModel = learnHiddenTAN(currentCardinality, parameterLearning,dataSet,randomChowLiuRoot, statisticalTest);

            // New model is better than previous model and the difference is greater than the threshold
            if(currentModel.getScoreValue() < bestModel.getScoreValue() && Math.abs(currentModel.getScoreValue() - bestModel.getScoreValue()) > threshold)
                bestModel = currentModel;
            else
                return bestModel;

        }

        // If the model wasn't returned before reached the maximum cardinality, the last iteration's model is returned
        // It is learned in case there wasn't even one iteration. A NB model with cardinality 2 would then be returned (the dumb model)
        return parameterLearning.learnModel(bestModel.getBayesianNetwork(), dataSet);
    }

    // Aprende un modelo TAN con variable latente para la cardinalidad actual
    private static LearningResult<DiscreteBayesNet> learnHiddenTAN (int cardinality,
                                                                    DiscreteParameterLearning parameterLearning,
                                                                    DiscreteData dataSet,
                                                                    boolean randomChowLiuRoot,
                                                                    DiscreteStatisticalTest statisticalTest) {

        // Creamos un modelo base el cual aprendemos con el EM para calcular su CL tree
        HLCM initialModel = HlcmCreator.createLCM(dataSet.getVariables(), cardinality);
        initialModel = (HLCM) parameterLearning.learnModel(initialModel, dataSet).getBayesianNetwork();

        // Estimate the Chow-Liu tree of its manifest variables conditioned on the root of the NB model
        WeightedUndirectedGraph<DiscreteVariable> clTree = ChowLiu.learnChowLiuTree(initialModel.getManifestVariables(),
                initialModel.getRoot().getVariable(), initialModel, dataSet, statisticalTest);
        if(randomChowLiuRoot)
            return learnRandomRootTAN(initialModel, parameterLearning, dataSet, clTree);
        else
            return learnBestRootTAN(initialModel, parameterLearning, dataSet, clTree);
    }

    private static LearningResult<DiscreteBayesNet> learnRandomRootTAN(HLCM initialModel,
                                                                       DiscreteParameterLearning parameterLearning,
                                                                       DiscreteData dataSet,
                                                                       WeightedUndirectedGraph<DiscreteVariable> chowLiuTree){

        // A random node is chosen to be the root of the Chow-Liu tree
        // Note: Do not confuse this root with the root of the Naive Bayes model.
        Random random = new Random();
        int rootIndex = random.nextInt(initialModel.getManifestNodes().size() - 1);

        return learnTAN(initialModel, parameterLearning, dataSet, chowLiuTree, rootIndex);
    }

    private static LearningResult<DiscreteBayesNet> learnBestRootTAN(HLCM initialModel,
                                                                     DiscreteParameterLearning parameterLearning,
                                                                     DiscreteData dataSet,
                                                                     WeightedUndirectedGraph<DiscreteVariable> chowLiuTree){

        LearningResult<DiscreteBayesNet> bestLearningResult = null;

        for(int i = 0; i < initialModel.getManifestNodes().size(); i++){

            // The model is cloned so it wont interfere with subsequent iterations
            HLCM clonedModel = initialModel.clone();

            LearningResult<DiscreteBayesNet> result = learnTAN(clonedModel, parameterLearning, dataSet, chowLiuTree, i);
            if(bestLearningResult == null || bestLearningResult.getScoreValue() < result.getScoreValue())
                bestLearningResult = result;
        }
        return  bestLearningResult;
    }

    private static LearningResult<DiscreteBayesNet> learnTAN(HLCM initialModel,
                                                             DiscreteParameterLearning parameterLearning,
                                                             DiscreteData dataSet,
                                                             WeightedUndirectedGraph<DiscreteVariable> chowLiuTree,
                                                             int rootIndex){

        // After that we have an undirected tree. To make it directed, a root node is used.
        AbstractNode<DiscreteVariable> root = chowLiuTree.getNodes().get(rootIndex);

        // Once the CL tree root has been chosen, a directed graph is created by recursively iterating through the graph
        List<AbstractNode<DiscreteVariable>> visitedNodes = new ArrayList<>();
        visitedNodes.add(root);
        DirectedAcyclicGraph<DiscreteVariable> directedClTree = new DirectedAcyclicGraph<>();
        directedClTree.addNode(root.getContent());
        iterateChildNodes(directedClTree, visitedNodes, root);

        // Now that the DAG has been filled, its edges are added to the Naive Bayes,
        // generating a Tree-agumented Naive Bayes model (TAN)
        for(Edge<DiscreteVariable> edge: directedClTree.getEdges()){
            initialModel.addEdge(initialModel.getNode(edge.getHead().getContent()),
                    initialModel.getNode(edge.getTail().getContent()));
        }

        // The TAN parameters are learned and the model is returned
        return parameterLearning.learnModel(initialModel, dataSet);
    }

    // tailRecursive method
    private static void iterateChildNodes(DirectedAcyclicGraph<DiscreteVariable> resultingGraph,
                                   List<AbstractNode<DiscreteVariable>> visitedNodes,
                                   AbstractNode<DiscreteVariable> node){

        for (AbstractNode<DiscreteVariable> neighbour: node.getNeighbors()){
            if(!visitedNodes.contains(neighbour)){

                // The neighbour node is set as visited
                visitedNodes.add(neighbour);

                // The new graph's nodes are required for adding the new edge between them
                DirectedNode<DiscreteVariable> toNeighbourNode = resultingGraph.addNode(neighbour.getContent()); // new -> added
                DirectedNode<DiscreteVariable> fromNode = resultingGraph.getNode(node.getContent()); // old -> retrieved

                resultingGraph.addEdge(toNeighbourNode, fromNode);
                iterateChildNodes(resultingGraph, visitedNodes, neighbour);
            }
        }
    }

}
