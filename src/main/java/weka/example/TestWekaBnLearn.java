package weka.example;

import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.learning.LearningResult;
import voltric.learning.parameter.mle.MLE;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;
import weka.bif.XmlBifReader;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.classifiers.bayes.net.search.local.K2;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 12/12/2017.
 */
public class TestWekaBnLearn {

    public static void main(String[] args) throws Exception{


        String asia_data = "data/Asia_data.arff";
        String webkb_data ="data/webkb/webkb_noclass.arff";
        String d897 = "data/parkinson/s6/d897_motor_nms55_s6.arff";
        String d897_motor = "data/parkinson/motor/d897_motor25.arff";
        String d897_nms = "data/parkinson/s6/nms/d897_nms30_s6.arff";
        String d897_noise = "data/parkinson/s6/d897_noise.arff";

        String dataString = asia_data;

        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(dataString));
        Instances dataWeka = loader.getDataSet();
        dataWeka.setClassIndex(0);

        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData(dataString);
/*
        DiscreteBayesNet xmlBifBn = testHC(dataWeka);
        LearningResult<DiscreteBayesNet> manualBnResult = createHC(dataVoltric);

        System.out.println("Manual BN result score ("+ manualBnResult.getScoreType()+") = " + manualBnResult.getScoreValue());
        System.out.println("Weka BN score (AIC): " + LearningScore.calculateAIC(dataVoltric, xmlBifBn));
*/
        testAsiaProjectedLearn(dataString);
    }

    private static void testAsiaProjectedLearn(String dataString) throws Exception{
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(dataString));
        Instances dataWeka = loader.getDataSet();

        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData(dataString);

        /** Creamos una lista con las variables de voltric que queremos proyectar */

        List<DiscreteVariable> projectionVariables = new ArrayList<>();
        projectionVariables.add(dataVoltric.getVariables().get(1));// vTuberculosis
        projectionVariables.add(dataVoltric.getVariables().get(4));// vTbOrCa

        List<String> projectionVariablesNames = projectionVariables.stream().map(DiscreteVariable::getName).collect(Collectors.toList());

        /** Definimos los parametros de filter utilizado */
        String projectionVarNamesString = "";
        for(String varName: projectionVariablesNames)
            projectionVarNamesString += varName + "|";

        String optionsString = "-E ^(?!("+projectionVarNamesString+")$).*$";
        String[] options = weka.core.Utils.splitOptions(optionsString);

        // Ejecutamos un bucle donde se aplican varios filtros consecutivos hasta que realmente filtre los datos bien
        int filteredAttributesSize = dataVoltric.getVariables().size();

        /** Eliminamos los atributos de dataWeka con weka.RemoveByName*/

        Instances projectedDataWeka = filterAttributesByName(dataWeka, options);

        while (projectedDataWeka.get(0).numAttributes() > projectionVariables.size()){
            projectedDataWeka =  filterAttributesByName(projectedDataWeka, options);
        }

        projectedDataWeka.setClassIndex(0);

        DiscreteBayesNet k2Bn = testK2(projectedDataWeka);
        DiscreteBayesNet hcBn = testHC(projectedDataWeka);

        int i = 0;
    }

    private static Instances filterAttributesByName(Instances dataWeka, String[] options) throws Exception{
        RemoveByName removeByNameFilter = new RemoveByName();
        removeByNameFilter.setOptions(options);
        removeByNameFilter.setInputFormat(dataWeka);
        Instances projectedDataWeka = Filter.useFilter(dataWeka, removeByNameFilter);
        return projectedDataWeka;
    }

    private static DiscreteBayesNet testK2(Instances data) throws Exception{

        System.out.println("========== K2 ===========");

        BayesNet emptyNet = new BayesNet();
        K2 oSearchAlgorithm = new K2();
        oSearchAlgorithm.setInitAsNaiveBayes(false);
        oSearchAlgorithm.setMaxNrOfParents(3);
        emptyNet.setSearchAlgorithm(oSearchAlgorithm);
        emptyNet.buildClassifier(data); // Learns a network from the empty net
        String bifNet = emptyNet.toXMLBIF03();

        DiscreteBayesNet xmlBifBn = XmlBifReader.processString(bifNet);
        return xmlBifBn;
    }

    private static DiscreteBayesNet testHC(Instances data) throws Exception{

        System.out.println("========== Hill-climbing ===========");

        BayesNet emptyNet = new BayesNet();
        HillClimber oSearchAlgorithm = new HillClimber();
        oSearchAlgorithm.setInitAsNaiveBayes(false);
        oSearchAlgorithm.setMaxNrOfParents(15);
        emptyNet.setSearchAlgorithm(oSearchAlgorithm);
        emptyNet.buildClassifier(data); // Learns a network from the empty net
        String bifNet = emptyNet.toXMLBIF03();

        DiscreteBayesNet xmlBifBn = XmlBifReader.processString(bifNet);
        return xmlBifBn;
    }

    private static LearningResult<DiscreteBayesNet> createHC(DiscreteData data){
        DiscreteVariable scm1rue = data.getVariables().get(0);
        DiscreteVariable scm1lue = data.getVariables().get(1);
        DiscreteVariable scm2rue = data.getVariables().get(2);
        DiscreteVariable scm2lue = data.getVariables().get(3);
        DiscreteVariable scm3rue = data.getVariables().get(4);
        DiscreteVariable scm3lue = data.getVariables().get(5);
        DiscreteVariable scm4rue = data.getVariables().get(6);
        DiscreteVariable scm4lue = data.getVariables().get(7);
        DiscreteVariable scm5rise = data.getVariables().get(8);
        DiscreteVariable scm6post = data.getVariables().get(9);
        DiscreteVariable scm7gait = data.getVariables().get(10);
        DiscreteVariable scm8spee = data.getVariables().get(11);
        DiscreteVariable scm9free = data.getVariables().get(12);
        DiscreteVariable scm10swa = data.getVariables().get(13);
        DiscreteVariable scm11spe = data.getVariables().get(14);
        DiscreteVariable scm12fee = data.getVariables().get(15);
        DiscreteVariable scm13dre = data.getVariables().get(16);
        DiscreteVariable scm14hyg = data.getVariables().get(17);
        DiscreteVariable scm15cha = data.getVariables().get(18);
        DiscreteVariable scm16wal = data.getVariables().get(19);
        DiscreteVariable scm17han = data.getVariables().get(20);
        DiscreteVariable scm18dpr = data.getVariables().get(21);
        DiscreteVariable scm19dsv = data.getVariables().get(22);
        DiscreteVariable scm20fpr = data.getVariables().get(23);
        DiscreteVariable scm21fsv = data.getVariables().get(24);

        DiscreteBayesNet bn = new DiscreteBayesNet("myBn");

        for(DiscreteVariable var: data.getVariables())
            bn.addNode(var);

        // scm1rue
        bn.addEdge(bn.getNode(scm1rue), bn.getNode(scm2rue));
        // scm1lue
        bn.addEdge(bn.getNode(scm1lue), bn.getNode(scm2lue));
        bn.addEdge(bn.getNode(scm1lue), bn.getNode(scm1rue));
        // scm2rue
        bn.addEdge(bn.getNode(scm2rue), bn.getNode(scm2lue));
        bn.addEdge(bn.getNode(scm2rue), bn.getNode(scm4rue));
        // scm2lue
        bn.addEdge(bn.getNode(scm2lue), bn.getNode(scm3lue));
        // scm3rue
        bn.addEdge(bn.getNode(scm3rue), bn.getNode(scm4rue));
        bn.addEdge(bn.getNode(scm3rue), bn.getNode(scm3lue));
        // scm3lue
        bn.addEdge(bn.getNode(scm3lue), bn.getNode(scm4lue));
        bn.addEdge(bn.getNode(scm3lue), bn.getNode(scm16wal));
        // scm4rue
        bn.addEdge(bn.getNode(scm4rue), bn.getNode(scm13dre));
        // scm4lue
        bn.addEdge(bn.getNode(scm4lue), bn.getNode(scm4rue));
        bn.addEdge(bn.getNode(scm4lue), bn.getNode(scm16wal));
        // scm5rise
        bn.addEdge(bn.getNode(scm5rise), bn.getNode(scm7gait));
        // scm6post
        bn.addEdge(bn.getNode(scm6post), bn.getNode(scm5rise));
        bn.addEdge(bn.getNode(scm6post), bn.getNode(scm7gait));
        // scm7gait
        bn.addEdge(bn.getNode(scm7gait), bn.getNode(scm8spee));
        // scm8spee
        bn.addEdge(bn.getNode(scm8spee), bn.getNode(scm11spe));
        // scm9free
        bn.addEdge(bn.getNode(scm9free), bn.getNode(scm7gait));
        bn.addEdge(bn.getNode(scm9free), bn.getNode(scm15cha));
        // scm11spe
        bn.addEdge(bn.getNode(scm11spe), bn.getNode(scm10swa));
        // scm12fee
        bn.addEdge(bn.getNode(scm12fee), bn.getNode(scm13dre));
        bn.addEdge(bn.getNode(scm12fee), bn.getNode(scm14hyg));
        // scm13dre
        bn.addEdge(bn.getNode(scm13dre), bn.getNode(scm14hyg));
        bn.addEdge(bn.getNode(scm13dre), bn.getNode(scm16wal));
        // scm14hyg
        bn.addEdge(bn.getNode(scm14hyg), bn.getNode(scm16wal));
        bn.addEdge(bn.getNode(scm14hyg), bn.getNode(scm8spee));
        // scm15cha
        bn.addEdge(bn.getNode(scm15cha), bn.getNode(scm13dre));
        bn.addEdge(bn.getNode(scm15cha), bn.getNode(scm16wal));
        //scm16wal
        bn.addEdge(bn.getNode(scm16wal), bn.getNode(scm7gait));
        bn.addEdge(bn.getNode(scm16wal), bn.getNode(scm6post));
        // scm17han
        bn.addEdge(bn.getNode(scm17han), bn.getNode(scm12fee));
        bn.addEdge(bn.getNode(scm17han), bn.getNode(scm11spe));
        bn.addEdge(bn.getNode(scm17han), bn.getNode(scm16wal));
        // scm18dpr
        bn.addEdge(bn.getNode(scm18dpr), bn.getNode(scm21fsv));
        // scm19dsv
        bn.addEdge(bn.getNode(scm19dsv), bn.getNode(scm18dpr));
        // scm20fpr
        bn.addEdge(bn.getNode(scm20fpr), bn.getNode(scm9free));
        bn.addEdge(bn.getNode(scm20fpr), bn.getNode(scm15cha));
        // scm21fsv
        bn.addEdge(bn.getNode(scm21fsv), bn.getNode(scm20fpr));

        MLE mle = new MLE(ScoreType.AIC);
        LearningResult<DiscreteBayesNet> bnResult = mle.learnModel(bn, data);
        return bnResult;
    }
}
