package experiment.scripts;

import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.model.DiscreteBayesNet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

/**
 * Created by equipo on 09/05/2018.
 *
 * Me da problemas los BIFs generados por BI e EAST, asi que como JBayes es capaz de leerlos, hago el siguiente proceso:
 *
 * BIF original -> XML (JBayes) -> BIF (Voltric)
 */
public class XmlToBnLearnBIF {

    public static void main(String[] args) throws Exception{

        // Cargamos la red en formato XML
        DiscreteBayesNet bn = XmlBifReader.processFile(new File("EAST_real_condvida_hlcm_2.xml"));

        // La exportamos en formato BIF (Voltric)
        OutputStream nbOutput = new FileOutputStream("EAST_real_condvida_2.bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(bn);
    }
}
