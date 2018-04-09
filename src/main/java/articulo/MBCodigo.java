package articulo;

import articulo.facet.DataFacet;
import articulo.facet.DataPartitionModel;
import voltric.clustering.util.AssignToClusters;
import voltric.clustering.util.ClusterValidation;
import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.graph.Edge;
import voltric.io.data.DataFileLoader;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.parameter.ParameterLearner;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.ScoreType;
import voltric.learning.structure.hillclimbing.GeneralHillClimbing;
import voltric.learning.structure.hillclimbing.operator.HcOperator;
import voltric.learning.structure.hillclimbing.operator.IncreaseLatentCardinality;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.util.SymmetricPair;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;
import voltric.variables.modelTypes.VariableType;
import weka.bif.XmlBifReader;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Codigo especifico para el metodo basado en la estimacion del Markov Blanket.
 *
 * TODO: Codigo articulo deberia quedarse como un conjunto de metodos generales
 */
public class MBCodigo extends Codigo{

    // El objetivo es definir una faceta con el markov blanket correspondiente a cada nodo
    public List<DataFacet> facetDeterminationWithMarkovBlanket(DiscreteData dataVoltric, Instances dataWeka) throws Exception {
        List<DataFacet> dataFacets = new ArrayList<>();

        DiscreteBayesNet bn = learnBnUsingWeka(dataVoltric.getVariables(), dataWeka);
        List<DiscreteBayesNet> mbs = MarkovBlanketGenerator.generate(bn);

        // Por cada una de los Markov Blankets creamos una DataFacet
        // TODO: Por ahora no calculamos la TC o entropia
        for(int i = 0; i < mbs.size(); i++) {
            dataFacets.add(new DataFacet(mbs.get(i).getVariables(), mbs.get(i), 0, 0, i));
        }

        return dataFacets;
    }

    // Este caso sirve por ejemplo cuando hemos aprendido la red utilizando bnLearn y la cargamos en Voltric
    public List<DataFacet> facetDeterminationWithMarkovBlanket(DiscreteBayesNet bayesNet) {
        List<DataFacet> dataFacets = new ArrayList<>();

        List<DiscreteBayesNet> mbs = MarkovBlanketGenerator.generate(bayesNet);

        // Por cada una de los Markov Blankets creamos una DataFacet
        // TODO: Por ahora no calculamos la TC o entropia
        for(int i = 0; i < mbs.size(); i++) {
            dataFacets.add(new DataFacet(mbs.get(i).getVariables(), mbs.get(i), 0, 0, i));
        }

        return dataFacets;
    }

    /* Aprende una particion por cada faceta correspondiente a un MB. El aprendizaje de la particion se realiza con un HKDB
        donde se mantiene la estructura entre MVs para que al generar el modelo multidimensional no se formen arcos imposibles.
     */
    public List<LearningResult<DiscreteBayesNet>> generatePartitions(List<DataFacet> argumentFacets, DiscreteData dataVoltric) throws Exception{

        List<LearningResult<DiscreteBayesNet>> individualClusteringSolutions = new ArrayList<>();

        // Por cada facet creamos un HLCM que corresponde una partición
        for(DataFacet facet: argumentFacets){
            LearningResult<DiscreteBayesNet> learningResult = learnHKDBwithoutMVchanges(facet.getBayesNet(),
                    dataVoltric, new ParallelEM(new EmConfig(), ScoreType.BIC), 10, 5.0, 500, "P_" + facet.getIndex());

            individualClusteringSolutions.add(learningResult);
        }
        return individualClusteringSolutions;
    }

    // Calcula los Brier Scores y completa los datos con las variables latentes de las soluciones clustering individuales correspondientes a cada faceta
    public SymmetricPair<List<DataPartitionModel>, DiscreteData> completeData(List<LearningResult<DiscreteBayesNet>> individualClusteringSolutions, DiscreteData dataVoltric) {
        List<DataPartitionModel> partitionModels = new ArrayList<>();
        Map<DiscreteDataInstance, List<Integer>> completedData = new HashMap<>();

        // Inicializamos el Map
        for(DiscreteDataInstance instance: dataVoltric.getInstances())
            completedData.put(instance, new ArrayList<>());

        // Por cada solucion clustering, calculamos su Brier score y generamos una columna de DataSet
        for(LearningResult<DiscreteBayesNet> clusteringSolution: individualClusteringSolutions){

            DiscreteBayesNet bn = clusteringSolution.getBayesianNetwork();

            // Completamos los datos para poder calcular el Brier Score de cada uno de ellas
            List<SymmetricPair<DiscreteDataInstance, double[]>> clusterAssingments = AssignToClusters.assignDataCaseToCluster(dataVoltric, bn);

            // Calculamos el Brier score
            int cardinality = clusterAssingments.get(0).getSecond().length;
            double normalizedBrierScore = ClusterValidation.calculateNormalizedUniformBrierScore(clusterAssingments, dataVoltric, cardinality);
            DataPartitionModel partitionModel = new DataPartitionModel(clusteringSolution, normalizedBrierScore);

            // Almacenamos la particion que contiene el Brier score y la red bayesiana
            partitionModels.add(partitionModel);

            // Creamos una columna nueva con los cluster Assignments
            for(SymmetricPair<DiscreteDataInstance, double[]> clustAssignment : clusterAssingments){

                // Calculamos el índice de valor máximo
                double maxVal = 0;
                int maxIndex = 0;
                for(int i = 0; i < clustAssignment.getSecond().length; i++)
                    if(clustAssignment.getSecond()[i] > maxVal) {
                        maxIndex = i;
                        maxVal = clustAssignment.getSecond()[i];
                    }

                // Asignamos dicho valor para la instancia en cuestion referente a la variable de clustering actual
                List<Integer> clustVarAssignments = completedData.get(clustAssignment.getFirst());
                clustVarAssignments.add(maxIndex);
            }
        }

        // Una vez completados los datos debemos generar un DataSet donde los atributos son todas las LVs y que nos sirve para poder calcular la BN de las LVs

        // Por cada variable latente correspondiente a una solución clustering, creamos una MV cuyo nombre coincide y que servirá para el nuevo DataSet
        List<DiscreteVariable> completedDataSetNewVars = new ArrayList<>();
        for(LearningResult<DiscreteBayesNet> individualClusteringSolution: individualClusteringSolutions){
            DiscreteVariable latentVar = individualClusteringSolution.getBayesianNetwork().getLatentVariables().get(0);
            DiscreteVariable newManifestVar = new DiscreteVariable(latentVar.getCardinality(), VariableType.MANIFEST_VARIABLE, latentVar.getName());
            completedDataSetNewVars.add(newManifestVar);
        }

        // Creamos una nueva lista con todas las variables, primero las MVs y luego las LVs (ahora son MV tmb)
        List<DiscreteVariable> completedDataSetVars = new ArrayList<>(dataVoltric.getVariables());
        completedDataSetVars.addAll(completedDataSetNewVars);
        DiscreteData completedDataSet = new DiscreteData(completedDataSetVars);

        // Añadimos las instancias correspondientes
        // Por cada instancia de los datos antiguos añadimos los valores de las MVs seguidos de las LVs
        for(DiscreteDataInstance oldInstance: dataVoltric.getInstances()){
            int[] oldData = oldInstance.getNumericValues();
            List<Integer> latentVarData = completedData.get(oldInstance);
            int[] newData = new int[oldData.length + latentVarData.size()];
            // Primero las MVs
            for(int i = 0; i < oldData.length; i++)
                newData[i] = oldData[i];
            // Despues las LVs
            for(int i = oldData.length; i < newData.length; i++)
                newData[i] = latentVarData.get(i);

            // Add to the new completed data a new data instance containing both the MVs and the LVs
            completedDataSet.add(new DiscreteDataInstance(newData), dataVoltric.getWeight(oldInstance));
        }

        return new SymmetricPair<>(partitionModels, completedDataSet);
    }

    // TODO: Evaluamos los datos según su Brier score y decidimos si es necesario filtrar ciertas particiones
    public SymmetricPair<List<DataPartitionModel>, DiscreteData> filterCompletedData(SymmetricPair<List<DataPartitionModel>, DiscreteData> pair, double brierScoreValue) {
        return pair;
    }

    // Exportamos los datos filtrados en formato ARFF (allVars y solo LVs (filtramos las MVs) -> 2 archivos)
    public void exportCompletedData(DiscreteData completedData, DiscreteData oldData, String completedDataPath, String onlyLVsDataPath) {
        DataFileLoader.saveDiscreteData(completedData, completedDataPath);
        // Filtramos las MVs
        List<DiscreteVariable> filteredCompleteDataVars = completedData.getVariables().stream().filter(x-> !oldData.getVariables().contains(x)).collect(Collectors.toList());
        DataFileLoader.saveDiscreteData(completedData.project(filteredCompleteDataVars), onlyLVsDataPath);
    }

    // Aprendemos una red Bayesiana a partir de los datos exportados (y la exportamos para su visualizacion con Weka DAG viewer)
    public DiscreteBayesNet learnLatentVarsBn(String onlyLVsDataPath, String outputBnPath) throws Exception{
        /** Load Weka data */
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(onlyLVsDataPath));
        Instances dataWeka = loader.getDataSet();

        BayesNet emptyNet = new BayesNet();
        HillClimber oSearchAlgorithm = new HillClimber();
        oSearchAlgorithm.setInitAsNaiveBayes(false);
        oSearchAlgorithm.setMaxNrOfParents(15);
        emptyNet.setSearchAlgorithm(oSearchAlgorithm);
        emptyNet.buildClassifier(dataWeka); // Learns a network from the empty net
        String bifNet = emptyNet.toXMLBIF03();

        PrintWriter out = new PrintWriter(new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(outputBnPath), "UTF8")));
        out.println(bifNet);
        out.close();

        return XmlBifReader.processString(bifNet);
    }

    // Generamos un modelo multi-particion con la nueva BN y las particiones por si queremos hacer inferencia sobre el mismo.
    public DiscreteBayesNet generateMultiPartitionModel(List<DataPartitionModel> partitionModels, DiscreteBayesNet latentVarsBn, DiscreteData fullData) {
        // En este caso simplemente tenemos que generar un modelo cuya estructura venga dada por las particiones y las red bayesiana
        // Como se trata de datos completos, fullData debe contener tanto las instancias de las LVs como las de las MVs

        // El primer paso es generar una red bayesiana y añadir todos los nodos de partitionModels, asi como sus arcos
        DiscreteBayesNet multiPartitionModel = new DiscreteBayesNet("multi-partition-model");

        for(DataPartitionModel partitionModel: partitionModels){
            List<DiscreteVariable> partitionModelVars = partitionModel.getBnResult().getBayesianNetwork().getVariables();
            // Primero añadimos las variables de la particion
            for(DiscreteVariable var: partitionModelVars)
                if(!multiPartitionModel.containsVar(var))
                    multiPartitionModel.addNode(var);

            List<Edge<Variable>> partitionModelEdges = partitionModel.getBnResult().getBayesianNetwork().getEdges();
            // Después añadimos sus arcos
            for(Edge<Variable> partitionEdge: partitionModelEdges){
                Variable edgeHead = partitionEdge.getHead().getContent();
                Variable edgeTail = partitionEdge.getTail().getContent();
                multiPartitionModel.addEdge(multiPartitionModel.getNode(edgeHead), multiPartitionModel.getNode(edgeTail));
            }
        }

        // Una vez añadidos los nodos y arcos de la partición, añadimos los edges de la latentVarsBn
        for(Edge<Variable> latentVarsBnEdge: latentVarsBn.getEdges()){
            Variable edgeHead = latentVarsBnEdge.getHead().getContent();
            Variable edgeTail = latentVarsBnEdge.getTail().getContent();
            multiPartitionModel.addEdge(multiPartitionModel.getNode(edgeHead), multiPartitionModel.getNode(edgeTail));
        }

        // Finalmente aprendemos con MLE los parametros del modelo (utilizando los datos completados de forma local con cada faceta)
        ParameterLearner.computeMLE(multiPartitionModel, fullData);

        return multiPartitionModel;
    }

    /*****************************************************************************************************************/

    private LearningResult<DiscreteBayesNet> learnHKDBwithoutMVchanges(DiscreteBayesNet seedtNet,
                                                                       DiscreteData dataVoltric,
                                                                       DiscreteParameterLearning parameterLearning,
                                                                       int maxCardinality,
                                                                       double threshold,
                                                                       int maxIterations,
                                                                       String latentVarName) {

        /** First a K-db model of cardinality 2 is created based on the seedNet's network */
        DiscreteBayesNet initialModel = new DiscreteBayesNet();
        DiscreteBeliefNode root;
        // The root variable is created
        if(!latentVarName.equals(""))
            root = initialModel.addNode(new DiscreteVariable(2, VariableType.LATENT_VARIABLE, latentVarName));
        else
            root = initialModel.addNode(new DiscreteVariable(2, VariableType.LATENT_VARIABLE));

        // All the manifest variables of the seedNet are added to the new BN and an edge from the root to them is created
        for(DiscreteVariable variable: seedtNet.getManifestVariables())
            initialModel.addEdge(initialModel.addNode(variable), root);

        // Al the edges of the seedNet are also added to the initial model
        for(Edge<Variable> edge: seedtNet.getEdges()){
            Variable edgeTail = edge.getTail().getContent();
            Variable edgeHead = edge.getHead().getContent();
            initialModel.addEdge(initialModel.getNode(edgeHead), initialModel.getNode(edgeTail));
        }

        // Finally the model is randomly parametrized
        initialModel.randomlyParameterize();

        /** A hill-climbing search process is applied where only the increaseCardinality operator is allowed*/
        IncreaseLatentCardinality ilcOperator = new IncreaseLatentCardinality(maxCardinality);
        Set<HcOperator> operatorSet = new HashSet<>();
        operatorSet.add(ilcOperator);

        GeneralHillClimbing hillClimbing = new GeneralHillClimbing(operatorSet, maxIterations, threshold);

        return hillClimbing.learnModel(initialModel, dataVoltric, parameterLearning);
    }
}
