package experiment;

/**
 * Created by equipo on 26/04/2018.
 *
 * Ejecuto varias veces el algoritmo BI y miro sus estructuras.
 * Aprendo un modelo para cada una de esas estructuras con el EM y hago la media de LL con 5 ejecuciones.
 * El tiempo de ejecucion no tiene en cuenta la fase EM, ya que  contamos con que este proceso se ha hecho en el BI
 *
 * Nota: Si de 5 ejecuciones del BI, todas las estructuras son iguales, no hago el proceso 25 veces, sino solo 5 (5_BI * 5_EM)
 *
 * TODO: Probar a cargar un BIF 0.1 con JBayes para intentar que no tengamos que hacer el modelo a mano
 */
public class ExperimentBI {
}
