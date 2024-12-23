package nn;

import nn.math.Matrix;

import static nn.math.SharedRnd.getRnd;

public interface Initializer {

    void initWeights(Matrix weights, int layer);


    // -----------------------------------------------------------------
    // --- A few predefined ones ---------------------------------------
    // -----------------------------------------------------------------
    class Random implements Initializer {

        private double min;
        private double max;

        public Random(double min, double max) {
            this.min = min;
            this.max = max;
        }

        @Override
        public void initWeights(Matrix weights, int layer) {
            double delta = max - min;
            weights.map(value -> min + getRnd().nextDouble() * delta);
        }
    }


    class XavierUniform implements Initializer {
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(6.0 / (weights.cols() + weights.rows()));
            weights.map(value -> (getRnd().nextDouble() - 0.5) * factor);
        }
    }

    class XavierNormal implements Initializer {
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = Math.sqrt(2.0 / (weights.cols() + weights.rows()));
            weights.map(value -> getRnd().nextGaussian() * factor);
        }
    }

    class LeCunUniform implements Initializer {
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(3.0 / weights.cols());
            weights.map(value -> (getRnd().nextDouble() - 0.5) * factor);
        }
    }

    class LeCunNormal implements Initializer {
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 1.0 / Math.sqrt(weights.cols());
            weights.map(value -> getRnd().nextGaussian() * factor);
        }
    }

    class HeUniform implements Initializer {
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(6.0 / weights.cols());
            weights.map(value -> (getRnd().nextDouble() - 0.5) * factor);
        }
    }

    class HeNormal implements Initializer {
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = Math.sqrt(2.0 / weights.cols());
            weights.map(value -> getRnd().nextGaussian() * factor);
        }
    }

}
