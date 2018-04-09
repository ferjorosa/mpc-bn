package weka.example;

import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.variables.DiscreteVariable;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Clase creada para probar que nuestro regex funciona correctamente y sirve para proyectar un DataSet de Weka correctamente
 */
public class TestWekaFilter {

    public static void main(String[] args) throws Exception{

        /** Cargamos un DataSet tanto en Weka como en voltric */

        String asia_data = "data/Asia_data.arff";
        String webkb_data ="data/webkb/webkb_noclass.arff";
        String d897 = "data/parkinson/s6/d897_motor_nms55_s6.arff";
        String d897_motor = "data/parkinson/motor/d897_motor25.arff";
        String d897_nms = "data/parkinson/s6/nms/d897_nms30_s6.arff";
        String d897_noise = "data/parkinson/s6/d897_noise.arff";

        String dataString = d897;

        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(dataString));
        Instances dataWeka = loader.getDataSet();
        dataWeka.setClassIndex(0);

        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData(dataString);

        /** Creamos una lista con las variables de voltric que queremos proyectar */

        List<DiscreteVariable> projectionVariables = selectProjectionVars(dataVoltric);
        List<String> projectionVariablesNames = projectionVariables.stream().map(DiscreteVariable::getName).collect(Collectors.toList());

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
        List<Attribute> projectedAttributes = Collections.list(projectedDataWeka.enumerateAttributes());
        projectedAttributes.forEach(x->System.out.println(x.name()));
    }

    private static List<DiscreteVariable> selectProjectionVars(DiscreteData data){

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

        DiscreteVariable d1_lightheaded = data.getVariables().get(25);
        DiscreteVariable d1_fainting = data.getVariables().get(26);
        DiscreteVariable d2_drowsiness = data.getVariables().get(27);
        DiscreteVariable d2_fatigue = data.getVariables().get(28);
        DiscreteVariable d2_insomnia = data.getVariables().get(29);
        DiscreteVariable d2_rls = data.getVariables().get(30);
        DiscreteVariable d3_loss_interest = data.getVariables().get(31);
        DiscreteVariable d3_loss_activities = data.getVariables().get(32);
        DiscreteVariable d3_anxiety = data.getVariables().get(33);
        DiscreteVariable d3_depression = data.getVariables().get(34);
        DiscreteVariable d3_flat_affect = data.getVariables().get(35);
        DiscreteVariable d3_loss_pleasure = data.getVariables().get(36);
        DiscreteVariable d4_hallucination = data.getVariables().get(37);
        DiscreteVariable d4_delusion = data.getVariables().get(38);
        DiscreteVariable d4_diplopia = data.getVariables().get(39);
        DiscreteVariable d5_loss_concentration = data.getVariables().get(40);
        DiscreteVariable d5_forget_explicit = data.getVariables().get(41);
        DiscreteVariable d5_forget_implicit = data.getVariables().get(42);
        DiscreteVariable d6_drooling = data.getVariables().get(43);
        DiscreteVariable d6_swallowing = data.getVariables().get(44);
        DiscreteVariable d6_constipation = data.getVariables().get(45);
        DiscreteVariable d7_urinary_urgency = data.getVariables().get(46);
        DiscreteVariable d7_urinary_frequency = data.getVariables().get(47);
        DiscreteVariable d7_nocturia = data.getVariables().get(48);
        DiscreteVariable d8_sex_drive = data.getVariables().get(49);
        DiscreteVariable d8_sex_dysfunction = data.getVariables().get(50);
        DiscreteVariable d9_unexplained_pain = data.getVariables().get(51);
        DiscreteVariable d9_taste_smell = data.getVariables().get(52);
        DiscreteVariable d9_weight_change = data.getVariables().get(53);
        DiscreteVariable d9_sweating  = data.getVariables().get(54);

        List<DiscreteVariable> projectionVars = new ArrayList<>();
        projectionVars.add(scm20fpr);
        projectionVars.add(d2_rls);
        projectionVars.add(scm3lue);

        return projectionVars;
    }
}
