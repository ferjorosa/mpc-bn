import articulo.learning.olcm.OlcmHillClimbing;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

/**
 * Created by equipo on 18/04/2018.
 */
public class OlcmLearningSixScript {

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

        // Definimos el Hill-climbing
        OlcmHillClimbing hillClimbing = new OlcmHillClimbing(100000, 2.0);

        // Aprendemos el modelo

        // Exportamos el modelo en formatos BIF y OBIF

    }
}
