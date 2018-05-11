package experiment;

import voltric.io.model.xmlbif.XmlBifReader;
import voltric.model.DiscreteBayesNet;
import voltric.util.distance.Hellinger;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 09/05/2018.
 */
public class TestHellingerPartitions {

    public static void main(String[] args) throws Exception {
/*
        System.out.println("\n\n EAST average hellinger distances");
        List<Double> eastHellinger = east();

        for(Double d: eastHellinger)
            System.out.println(d + "\n");
*/

/*
        System.out.println("\n\n BI average hellinger distances");
        List<Double> biHellinger = bi();

        for(Double d: biHellinger)
            System.out.println(d + "\n");
*/


        System.out.println("\n\n OLHC average hellinger distances");
        List<Double> olhcHellinger = olhc();

        for(Double d: olhcHellinger)
            System.out.println(d + "\n");


/*
        System.out.println("\n\n LCM average hellinger distance");
        double lcmHellinger = lcm();
        System.out.println(lcmHellinger + "\n");

*/
    }

    private static double lcm() throws Exception {
        List<String> latentVars = new ArrayList<>();
        latentVars.add("variable218");

        // Cargamos el modelo del cual vamos a calcular su matriz de distancias de Hellinger y su distancia media por particion
        DiscreteBayesNet model = XmlBifReader.processFile(new File("LCM_condvida_3.xml"), latentVars);

        return Hellinger.averageClusterDistancesLCM(model);
    }

    private static List<Double> bi() throws Exception {
        List<String> latentVars = new ArrayList<>();
        latentVars.add("variable76");
        latentVars.add("variable77");
        latentVars.add("variable78");

        // Cargamos el modelo del cual vamos a calcular su matriz de distancias de Hellinger y su distancia media por particion
        DiscreteBayesNet model = XmlBifReader.processFile(new File("BI_real_condvida_4.xml"), latentVars);

        return Hellinger.averageClusterDistances(model);
    }

    private static List<Double> east() throws Exception {
        List<String> latentVars = new ArrayList<>();
        latentVars.add("variable26");
        latentVars.add("variable12");
        latentVars.add("variable14");
        latentVars.add("variable24");

        // Cargamos el modelo del cual vamos a calcular su matriz de distancias de Hellinger y su distancia media por particion
        DiscreteBayesNet east_model = XmlBifReader.processFile(new File("EAST_real_condvida_2.xml"), latentVars);

        return Hellinger.averageClusterDistances(east_model);
    }

    private static List<Double> olhc() throws Exception{
        List<String> latentVars = new ArrayList<>();
        latentVars.add("variable498");
        latentVars.add("variable119");
        latentVars.add("variable531");
        latentVars.add("variable681");

        // Cargamos el modelo del cual vamos a calcular su matriz de distancias de Hellinger y su distancia media por particion
        DiscreteBayesNet olhc_model = XmlBifReader.processFile(new File("OLHC_real_condvida_1.xml"), latentVars);

        return Hellinger.averageClusterDistances(olhc_model);
    }
}
