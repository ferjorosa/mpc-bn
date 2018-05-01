package articulo.learning;

import articulo.learning.olcm.operator.*;
import voltric.data.DiscreteData;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.bif.OldBifFileWriter;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.structure.hillclimbing.operator.HcOperator;
import voltric.model.DiscreteBayesNet;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by equipo on 16/04/2018.
 *
 * Algoritmo Hill-climbing diseñado para trabajar con modelos OLCM. Se compone de 2 fases:
 * - Expansion de la estructura
 * - Simplificacion de la estructura
 *
 * Toma inspiracion de los metodos de aprendizaje de Zhang y Chen.
 *
 * TODO: Quizas es interesante utilizarla estrategia de combinada de ambos para separar la cardinalidad de la "estructura" , lo cual tmb lo hace Liu
 * TODO: Probar con 2 estructuras de inicio:
 * 1 - Un LCM
 * 2 - Una version OLCM sin overlapping y como el overlapping mejora la estructura
 */
public class OlcmHillClimbing {

    private Set<OlcmHcOperator> expansionOperators;

    private Set<OlcmHcOperator> simplificationOperators;

    private int maxIterations;

    private double threshold;

    public OlcmHillClimbing(int maxIterations,
                            double threshold,
                            Set<OlcmHcOperator> expansionOperators,
                            Set<OlcmHcOperator> simplificationOperators){
        /** Expansion Operators */
        this.expansionOperators =expansionOperators;

        /** Simplification Operators */
        this.simplificationOperators =simplificationOperators;

        this.maxIterations = maxIterations;
        this.threshold = threshold;
    }

    public LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet seedNet, DiscreteData data, String dataName, AbstractEM em) throws Exception{

        LearningResult<DiscreteBayesNet> previousIterationResult = em.learnModel(seedNet.clone(), data);

        int iterations = 0;

        // Bucle general que itera por las fases de expansin y simplificacion
        while(iterations < this.maxIterations) {
            iterations = iterations + 1;
            System.out.println("iterations ("+ iterations + ") score: " + previousIterationResult.getScoreValue());

            /** Store current model in case it is needed afterwards */
            storeModel(previousIterationResult.getBayesianNetwork(), dataName, iterations);

            /** Expansion process */
            LearningResult<DiscreteBayesNet> expansionBestResult = expansionProcess(previousIterationResult, data, em);

            // Store expansion model
            storeIntermidiateModel(expansionBestResult.getBayesianNetwork(), dataName, "expansion", iterations);

            // If the expansion process haven't increased the score, we stop the HC algorithm
            if(previousIterationResult.getScoreValue() >= expansionBestResult.getScoreValue()
                    || Math.abs(expansionBestResult.getScoreValue() - previousIterationResult.getScoreValue()) < threshold)
                return previousIterationResult;

            /** Simplification process */
            LearningResult<DiscreteBayesNet> simplificationBestResult = simplificationProcess(expansionBestResult, data, em);

            // Store simplification model
            storeIntermidiateModel(expansionBestResult.getBayesianNetwork(), dataName, "simplification", iterations);

            // If the simplification process haven't increased the score, we stop the HC algorithm
            if(expansionBestResult.getScoreValue() >= simplificationBestResult.getScoreValue()
                    || Math.abs(expansionBestResult.getScoreValue() - simplificationBestResult.getScoreValue()) < threshold)
                return expansionBestResult;

            // If both the expansion and simplification processes have improved the model's score, the model is stored in memory
            previousIterationResult = simplificationBestResult;
        }

        // Si se ha pasado el maximo de iteraciones y segui mejorando el score, lo corto el HC y devuelvo el modelo
        return previousIterationResult;
    }

    public  LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {

        LearningResult<DiscreteBayesNet> previousIterationResult = em.learnModel(seedNet.clone(), data);

        int iterations = 0;

        // Bucle general que itera por las fases de expansin y simplificacion
        while (iterations < this.maxIterations) {
            iterations = iterations + 1;
            System.out.println("iterations (" + iterations + ") score: " + previousIterationResult.getScoreValue());

            /** Expansion process */
            LearningResult<DiscreteBayesNet> expansionBestResult = expansionProcess(previousIterationResult, data, em);

            // If the expansion process haven't increased the score, we stop the HC algorithm
            if (previousIterationResult.getScoreValue() >= expansionBestResult.getScoreValue()
                    || Math.abs(expansionBestResult.getScoreValue() - previousIterationResult.getScoreValue()) < threshold)
                return previousIterationResult;

            /** Simplification process */
            LearningResult<DiscreteBayesNet> simplificationBestResult = simplificationProcess(expansionBestResult, data, em);

            // If the simplification process haven't increased the score, we stop the HC algorithm
            if (expansionBestResult.getScoreValue() >= simplificationBestResult.getScoreValue()
                    || Math.abs(expansionBestResult.getScoreValue() - simplificationBestResult.getScoreValue()) < threshold)
                return expansionBestResult;

            // If both the expansion and simplification processes have improved the model's score, the model is stored in memory
            previousIterationResult = simplificationBestResult;
        }

        // Si se ha pasado el maximo de iteraciones y segui mejorando el score, lo corto el HC y devuelvo el modelo
        return previousIterationResult;
    }

    private LearningResult<DiscreteBayesNet> expansionProcess (final LearningResult<DiscreteBayesNet> previousIterationResult, DiscreteData data, AbstractEM em) {

        LearningResult<DiscreteBayesNet> expansionBestResult = previousIterationResult;
        LearningResult<DiscreteBayesNet> previousResult;

        do {
            System.out.println("Expansion score: " + expansionBestResult.getScoreValue());

            previousResult = expansionBestResult;
            for (OlcmHcOperator operator : this.expansionOperators) {
                LearningResult<DiscreteBayesNet> expansionResult = operator.apply(expansionBestResult.getBayesianNetwork(), data, em);
                if (expansionResult.getScoreValue() > expansionBestResult.getScoreValue()) {
                    expansionBestResult = expansionResult;
                }
            }
        } while (expansionBestResult.getScoreValue() > previousResult.getScoreValue());

        return expansionBestResult;
    }

    private LearningResult<DiscreteBayesNet> simplificationProcess (final LearningResult<DiscreteBayesNet> expansionBestResult, DiscreteData data, AbstractEM em) {

        LearningResult<DiscreteBayesNet> simplificationBestResult = expansionBestResult;
        LearningResult<DiscreteBayesNet> previousResult;

        do {
            System.out.println("Simplification score: " + simplificationBestResult.getScoreValue());

            previousResult = simplificationBestResult;
            for (OlcmHcOperator operator : this.simplificationOperators) {
                LearningResult<DiscreteBayesNet> simplificationResult = operator.apply(simplificationBestResult.getBayesianNetwork(), data, em);
                if (simplificationResult.getScoreValue() > simplificationBestResult.getScoreValue()) {
                    simplificationBestResult = simplificationResult;
                }
            }
        } while (simplificationBestResult.getScoreValue() > previousResult.getScoreValue());

        return simplificationBestResult;
    }

    private void storeModel(DiscreteBayesNet model, String dataName, int iterationNumber) throws Exception{
        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream(dataName+"/olcmHC/olcm_iter_"+iterationNumber+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(model);

        // Export model in OBIF format
        //OldBifFileWriter.writeBif("estudios/"+dataName+"/olcmHC/olcm_iter_"+iterationNumber+".obif", model);
    }

    private void storeIntermidiateModel(DiscreteBayesNet model, String dataName, String process, int iterationNumber) throws Exception{
        // Export model in BIF 0.15 format
        OutputStream nbOutput = new FileOutputStream(dataName+"/olcmHC/intermidiate/olcm_"+process+"_iter_"+iterationNumber+".bif");
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(nbOutput);
        writer.write(model);

        // Export model in OBIF format
        //OldBifFileWriter.writeBif("estudios/"+dataName+"/olcmHC/intermidiate/olcm_"+process+"_iter_"+iterationNumber+".obif", model);
    }
}
