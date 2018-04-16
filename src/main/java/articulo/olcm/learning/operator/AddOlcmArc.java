package articulo.olcm.learning.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;

/**
 * Created by equipo on 16/04/2018.
 *
 * Adds a non-repeated arc between a LV and a MV, forming a OLCM
 */
public class AddOlcmArc implements OlcmHcOperator{
    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {
        return null;
    }
}
