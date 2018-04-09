package articulo.scripts;

import articulo.MBCodigo;
import articulo.facet.DataFacet;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.model.DiscreteBayesNet;
import weka.bif.XmlBifReader;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.List;

/**
 * Created by equipo on 09/04/2018.
 */
public class AttributeClusteringScript {

    public static void main(String[] args) throws Exception {

        // Parkinson
        String d897 = "data/parkinson/s6/d897_motor_nms55_s6.arff";
        String d897_motor = "data/parkinson/motor/d897_motor25.arff";
        String d897_nms = "data/parkinson/s6/nms/d897_nms30_s6.arff";
        String d897_noise = "data/parkinson/s6/d897_noise.arff";

        // Beeps
        String beeps2013 = "articulo/data/real/beeps/beeps2013_sampled_training_updated.arff";

        // Condiciones_vida
        String condiciones_vida = "articulo/data/real/condiciones_vida/condiciones_vida_2016_training.arff";

        // MENA - Egypt 2013
        String egypt2013 = "articulo/data/real/mena-egypt/Egypt_1927_training.arff";

        String dataString = condiciones_vida;
        String dataName = "condiciones_vida";

        /** Load Weka data */
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(dataString));
        Instances dataWeka = loader.getDataSet();
        dataWeka.setClassIndex(0);

        /** Load Voltric data */
        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData(dataString);

        MBCodigo codigo = new MBCodigo();

        /** Facets generated from a loaded BN */
        DiscreteBayesNet loadedBn = XmlBifReader.processFile(new File("estudios/condiciones_vida/condiciones_vida_2016.xml"));
        List<DataFacet> facets = codigo.facetDeterminationWithMarkovBlanket(loadedBn);

        /** LCMs genearated from data facets */

    }
}
