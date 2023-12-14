package nn.optimizer;

import nn.math.Matrix;
import nn.math.Vec;

public interface Optimizer {

    void updateWeights(Matrix weights, Matrix dCdW);
    Vec updateBias(Vec bias, Vec dCdB);
    Optimizer copy();

}
