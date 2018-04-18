package articulo.learning.olcm.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 16/04/2018.
 *
 * Elimina el nodo latente entre LVs que tenga todos sus hijos con 2 o mas padres.
 *
 * TODO: si bien en cada iteracion seria mas facil volver a clonar la seedNet, añado el nodo en cuestion eliminado asi como sus hijos para no tener que copiar todo
 *
 * TODO: Quizas es mejor sustituir el comportamiento de este operador
 */
public class RemoveOlcmNode implements OlcmHcOperator{

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {

        // The BN is copied to avoid modifying current object.
        DiscreteBayesNet clonedNet = seedNet.clone();

        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        // Iteramos por el conjunto de los nodos latentes y probamos a eliminar dicho nodo siempre y cuando
        // todas las MVs afectadas tengan 2 o mas padres
        List<DiscreteBeliefNode> filteredLatentNodes = clonedNet.getLatentNodes().stream()
                .filter(x ->{
                    List<Boolean> childrenWithTwoOrMoreParents = x.getChildrenNodes().stream()
                            .map(y-> y.getParents().size() >= 2)
                            .collect(Collectors.toList());
                    return childrenWithTwoOrMoreParents.contains(false);
                }).collect(Collectors.toList());

        if(filteredLatentNodes.size() >= 1){

            DiscreteVariable currentlyRemovedNodeVariable;
            List<Variable> currentlyRemovedNodeChildren = new ArrayList<>();

            // Eliminamos el nodo y almacenamos el score del nuevo modelo
            for(DiscreteBeliefNode latentNode: filteredLatentNodes) {
                // Primero almacenamos tanto el nodo como sus hijos
                currentlyRemovedNodeChildren = latentNode.getChildrenNodes().stream().map(x-> x.getVariable()).collect(Collectors.toList());
                currentlyRemovedNodeVariable = latentNode.getVariable();

                // Despues lo eliminamos
                clonedNet.removeNode(latentNode);

                // Por ultimo aprendemos el modelo con el EM y lo almacenamos junto a su score si mejora el mejor modelo actual
                LearningResult<DiscreteBayesNet> removedLatentVarResult = em.learnModel(clonedNet, data);
                if (removedLatentVarResult.getScoreValue() > bestModelScore) {
                    bestModelScore = removedLatentVarResult.getScoreValue();
                    bestModelResult = removedLatentVarResult;
                }

                // Independientemente de si el nuevo nodo ha mejorado el score o no, reañadimos lo borrado
                DiscreteBeliefNode currentlyRemovedNode = clonedNet.addNode(currentlyRemovedNodeVariable);
                for(Variable child: currentlyRemovedNodeChildren)
                    clonedNet.addEdge(clonedNet.getNode(child), currentlyRemovedNode);

            }
        }

        if(bestModelResult != null)
            return bestModelResult;

        return new LearningResult<>(null, bestModelScore, em.getScoreType());
    }
}
